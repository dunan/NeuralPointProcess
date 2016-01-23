#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "dense_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "node_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "exp_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "simple_node_layer.h"
#include "batch_norm_param.h"
#include "config.h"
#include "data_loader.h"

const MatMode mode = GPU;

const char *datafile, *save_dir = "./saved";

GraphNN<mode, Dtype> net_train, net_test;
GraphData<mode, Dtype>* g_last_hidden_train, *g_last_hidden_test;
std::vector< GraphData<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;
LinearParam<mode, Dtype>* i2h, *h2h, *h2o;

ILayer<mode, Dtype>* AddNetBlocks(int time_step, GraphNN<mode, Dtype>& gnn, ILayer<mode, Dtype> *last_hidden_layer)
{    
    gnn.AddLayer(last_hidden_layer);
    
    auto* input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("input_%d", time_step));
    auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    hidden_layer->AddParam(input_layer->name, i2h);
    hidden_layer->AddParam(last_hidden_layer->name, h2h);                
    auto* relu_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("reluact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);
    auto* output_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("out_%d", time_step), h2o);
    auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);                                
        
    auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step));
    auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N);    	
            	
    gnn.AddLayer(input_layer);
    gnn.AddLayer(hidden_layer);
    gnn.AddLayer(relu_layer);        
    gnn.AddLayer(output_layer);
    gnn.AddLayer(exp_layer);
    gnn.AddLayer(mse_criterion);
        
    gnn.AddEdge(input_layer, hidden_layer);
    gnn.AddEdge(last_hidden_layer, hidden_layer);
    gnn.AddEdge(hidden_layer, relu_layer);
    gnn.AddEdge(relu_layer, output_layer);
    gnn.AddEdge(output_layer, exp_layer);        
    gnn.AddEdge(exp_layer, mse_criterion);    
    gnn.AddEdge(exp_layer, mae_criterion);
        
    return relu_layer;   
}

void InitNetTrain()
{    
    g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_train->node_states->DenseDerived().Zeros(cfg::batch_size, cfg::n_hidden);
    g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_test->node_states->DenseDerived().Zeros(cfg::batch_size, cfg::n_hidden);

    g_event_input.clear();
    g_event_label.clear();
    g_time_input.clear();
    g_time_label.clear();
            
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {
        g_event_input.push_back(new GraphData<mode, Dtype>(SPARSE));
        g_event_label.push_back(new GraphData<mode, Dtype>(SPARSE));
        g_time_input.push_back(new GraphData<mode, Dtype>(DENSE));
        g_time_label.push_back(new GraphData<mode, Dtype>(DENSE));        
    }
        
    i2h = new LinearParam<mode, Dtype>("i2h", 1, cfg::n_hidden, 0, 0.01);
    h2h = new LinearParam<mode, Dtype>("h2h", cfg::n_hidden, cfg::n_hidden, 0, 0.01);
    h2o = new LinearParam<mode, Dtype>("h2o", cfg::n_hidden, 1, 0, 0.01);
    
    net_train.AddParam(i2h);
    net_train.AddParam(h2h);
    net_train.AddParam(h2o);
    
    ILayer<mode, Dtype>* last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden_train");    
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {
        auto* new_hidden = AddNetBlocks(i, net_train, last_hidden_layer);
        last_hidden_layer = new_hidden;
    }
        
    net_test.AddParam(i2h);
    net_test.AddParam(h2h);
    net_test.AddParam(h2o);        
    auto* test_last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden_test");
    AddNetBlocks(0, net_test, test_last_hidden_layer);            
}

template<typename data_type> 
void ReadDataSeq(const char* datafile, std::vector<data_type>& raw_data)
{
    raw_data.clear();
    std::ifstream raw_stream(datafile);
    data_type buf;
    while (raw_stream >> buf)
    {
        raw_data.push_back(buf);
    }
}

void ReadRaw()
{
    std::vector<Dtype> time_data;
    std::vector<int> event_data;

    ReadDataSeq(cfg::f_time_data, time_data);
    ReadDataSeq(cfg::f_event_data, event_data);
    
    for (unsigned i = 0; i < event_data.size(); ++i)
        event_data[i]--;

    int data_len = time_data.size() - time_data.size() % (cfg::batch_size * cfg::bptt);    
    int seg_len = data_len / cfg::batch_size;
        
    int num_seg = seg_len / cfg::bptt;
    int test_len = (int)(num_seg * 0.1) * cfg::bptt;
    int train_len = seg_len - test_len;
        
    for (int i = time_data.size() - 1; i >= 1; --i)
        time_data[i] = time_data[i] - time_data[i - 1];
        
    train_data = new DataLoader<TRAIN>(1, cfg::batch_size); 
    test_data = new DataLoader<TEST>(1, cfg::batch_size);

    Dtype* time_data_ptr = time_data.data();
    int* event_data_ptr = event_data.data();
    for (unsigned i = 0; i < cfg::batch_size; ++i)
    {
        train_data->InsertSequence(event_data_ptr, time_data_ptr, train_len);
        event_data_ptr += train_len;
        time_data_ptr += train_len;
        test_data->InsertSequence(event_data_ptr, time_data_ptr, test_len);
        event_data_ptr += test_len;
        time_data_ptr += test_len;
    }
}

int main(const int argc, const char** argv)
{	
    cfg::LoadParams(argc, argv);    
	GPUHandle::Init(cfg::dev_id);
    
    ReadRaw();    
    InitNetTrain();
    int max_iter = (long long)cfg::max_epoch * train_data->num_samples / cfg::bptt / cfg::batch_size;
    int init_iter = cfg::iter;
    if (init_iter > 0)
	{
		printf("loading model for iter=%d\n", init_iter);
		net_train.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}
    
    Dtype mae, rmse;
    auto& last_hidden_train = g_last_hidden_train->node_states->DenseDerived();
    last_hidden_train.Zeros(cfg::batch_size, cfg::n_hidden);

    std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label;
    train_feat["last_hidden_train"] = g_last_hidden_train;
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {        
        train_feat[fmt::sprintf("input_%d", i)] = g_time_input[i];
        train_label[fmt::sprintf("mse_%d", i)] = g_time_label[i];
        train_label[fmt::sprintf("mae_%d", i)] = g_time_label[i];
    }
    
    for (; cfg::iter <= max_iter; ++cfg::iter)
	{
		if (cfg::iter % cfg::test_interval == 0)
		{
			std::cerr << "testing" << std::endl;
            rmse = mae = 0.0;
            auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived(); 
            last_hidden_test.Zeros(cfg::batch_size, cfg::n_hidden);

            test_data->StartNewEpoch();            
            while (test_data->NextBatch(g_last_hidden_test, 
                                        g_event_input[0], 
                                        g_time_input[0], 
                                        g_event_label[0], 
                                        g_time_label[0]))
            {
                net_test.ForwardData({{"input_0", g_time_input[0]}, {"last_hidden_test", g_last_hidden_test}}, TEST);
                auto loss_map = net_test.ForwardLabel({{"mse_0", g_time_label[0]}, {"mae_0", g_time_label[0]}});
                rmse += loss_map["mse_0"];
                mae += loss_map["mae_0"];
                net_test.GetDenseNodeState("reluact_0", last_hidden_test);
            }
            rmse = sqrt(rmse / test_data->num_samples); 
            mae /= test_data->num_samples;
			std::cerr << fmt::sprintf("test mae: %.4f\t test rmse: %.4f", mae, rmse) << std::endl;            
		}
        
        train_data->NextBpttBatch(cfg::bptt, 
                                  g_last_hidden_train, 
                                  g_event_input, 
                                  g_time_input, 
                                  g_event_label, 
                                  g_time_label);
        net_train.ForwardData(train_feat, TRAIN);        
        auto loss_map = net_train.ForwardLabel(train_label);
        //net_train.GetDenseNodeState(fmt::sprintf("reluact_%d", cfg::bptt - 1), last_hidden_train);

        net_train.BackPropagation();
        net_train.UpdateParams(cfg::lr, cfg::l2_penalty, cfg::momentum);    

        if (cfg::iter % cfg::report_interval == 0)
		{
            mae = rmse = 0.0;
            for (unsigned i = 0; i < cfg::bptt; ++i)
            {
                mae += loss_map[fmt::sprintf("mae_%d", i)];
                rmse += loss_map[fmt::sprintf("mse_%d", i)];  
            }
            rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
			mae /= cfg::bptt * cfg::batch_size;
			std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f", cfg::iter, mae, rmse) << std::endl;
/*            std::cerr << fmt::sprintf("d_i2h=%.6f\td_h2h: %.6f\td_h2o: %.6f", 
                                        i2h->delta_weight.Amax(), h2h->delta_weight.Amax(), h2o->delta_weight.Amax()) << std::endl;*/ 
		}        	
	}
    
	GPUHandle::Destroy();
	return 0;
}
