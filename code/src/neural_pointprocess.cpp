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
#include "classnll_criterion_layer.h"
#include "simple_node_layer.h"

#include "config.h"
#include "data_loader.h"

std::vector< std::vector<int> > raw_event_data;
std::vector< std::vector<Dtype> > raw_time_data;

template<typename data_type>
inline void LoadRaw(const char* filename, std::vector< std::vector<data_type> >& raw_data)
{
    raw_data.clear();
    std::ifstream f_stream(filename);
    std::string read_buf;
    data_type d;
    while (getline(f_stream, read_buf))
    {
        std::stringstream ss(read_buf);
        std::vector<data_type> cur_seq;
        cur_seq.clear();
        while (ss >> d)
        {
            cur_seq.push_back(d); 
        }
        raw_data.push_back(cur_seq);
    }       
}

inline void ReadRawData()
{
    std::cerr << "loading data..." << std::endl;
    assert(cfg::f_time_data && cfg::f_event_data);    
    
    LoadRaw(cfg::f_event_data, raw_event_data);
    LoadRaw(cfg::f_time_data, raw_time_data);
    assert(raw_event_data.size() == raw_time_data.size());
    std::set<int> label_set;
    label_set.clear();
    for (unsigned i = 0; i < raw_event_data.size(); ++i)
    {
        for (unsigned j = 0; j < raw_event_data[i].size(); ++j)
            label_set.insert(raw_event_data[i][j]);
    }
    std::cerr << "totally " << label_set.size() << " events" << std::endl;
    train_data = new DataLoader<TRAIN>(label_set.size(), cfg::batch_size); 
    test_data = new DataLoader<TEST>(label_set.size(), cfg::batch_size);
    
    for (unsigned i = 0; i < raw_event_data.size(); ++i)
    {
        assert(raw_event_data[i].size() == raw_time_data[i].size());
        
        int origin_len = raw_event_data[i].size();
        int test_len = origin_len * 0.1;
        int train_len = origin_len - test_len;
        for (int j = 0; j < origin_len; ++j)
            raw_event_data[i][j]--; // the raw event is 1-based
        for (int j = origin_len - 1; j >= 1; --j)
            raw_time_data[i][j] = raw_time_data[i][j] - raw_time_data[i][j-1];            
        train_data->InsertSequence(raw_event_data[i].data(), raw_time_data[i].data(), train_len);
        test_len++;
        test_data->InsertSequence(raw_event_data[i].data() + train_len - 1, raw_time_data[i].data() + train_len - 1, test_len);                             
    }
    std::cerr << raw_event_data.size() << " sequences loaded." << std::endl;
}

template<MatMode mode>
ILayer<mode, Dtype>* AddNetBlocks(int time_step, GraphNN<mode, Dtype>& gnn, ILayer<mode, Dtype> *last_hidden_layer, 
                                    std::map< std::string, LinearParam<mode, Dtype>* >& param_dict)
{
    gnn.AddLayer(last_hidden_layer);
    auto* event_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("event_input_%d", time_step));
    auto* time_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step));

    auto* embed_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("embed_%d", time_step), param_dict["w_embed"]); 

    auto* relu_embed_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_embed_%d", time_step), WriteType::INPLACE, ActTarget::NODE);

    auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    hidden_layer->AddParam(time_input_layer->name, param_dict["w_time2h"]); 
    hidden_layer->AddParam(relu_embed_layer->name, param_dict["w_event2h"]); 
    hidden_layer->AddParam(last_hidden_layer->name, param_dict["w_h2h"]); 

    auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_hidden_%d", time_step), WriteType::INPLACE, ActTarget::NODE);
    auto* event_output_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("event_out_%d", time_step), param_dict["w_event_out"]); 

    auto* time_out_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"]); 
    //auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);

    auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>(fmt::sprintf("nll_%d", time_step), true);
    auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step));
    auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N);

    typedef typename std::map< std::string, LinearParam<mode, Dtype>* >::iterator param_iter;
    for (param_iter it = param_dict.begin(); it != param_dict.end(); ++it)
        gnn.AddParam(it->second);

    gnn.AddLayer(event_input_layer);
    gnn.AddLayer(time_input_layer);
    gnn.AddLayer(embed_layer);
    gnn.AddLayer(relu_embed_layer);
    gnn.AddLayer(hidden_layer);
    gnn.AddLayer(relu_hidden_layer);
    gnn.AddLayer(event_output_layer);
    gnn.AddLayer(time_out_layer);
    //gnn.AddLayer(exp_layer);
    gnn.AddLayer(classnll);
    gnn.AddLayer(mse_criterion);
    gnn.AddLayer(mae_criterion);

    gnn.AddEdge(event_input_layer, embed_layer);
    gnn.AddEdge(embed_layer, relu_embed_layer);
    gnn.AddEdge(time_input_layer, hidden_layer);
    gnn.AddEdge(relu_embed_layer, hidden_layer);
    gnn.AddEdge(last_hidden_layer, hidden_layer);
    gnn.AddEdge(hidden_layer, relu_hidden_layer);
    gnn.AddEdge(relu_hidden_layer, event_output_layer);
    gnn.AddEdge(relu_hidden_layer, time_out_layer);
    //gnn.AddEdge(time_out_layer, exp_layer);

    gnn.AddEdge(event_output_layer, classnll);
    gnn.AddEdge(time_out_layer, mse_criterion);
    gnn.AddEdge(time_out_layer, mae_criterion);

    return relu_hidden_layer; 
}

template<MatMode mode>
void InitNet(GraphNN<mode, Dtype>& gnn, std::map< std::string, LinearParam<mode, Dtype>* >& param_dict, unsigned n_unfold)
{
    ILayer<mode, Dtype>* last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden");    

    for (unsigned i = 0; i < n_unfold; ++i)
    {
        auto* new_hidden = AddNetBlocks(i, gnn, last_hidden_layer, param_dict);
        last_hidden_layer = new_hidden;
    }
}

template<MatMode mode>
void Work()
{
    std::vector< GraphData<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;
     
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

    GraphNN<mode, Dtype> net_train, net_test;

    std::map< std::string, LinearParam<mode, Dtype>* > param_dict; 
    param_dict["w_embed"] = new LinearParam<mode, Dtype>("w_embed",  train_data->num_events, cfg::n_embed, 0, 0.01);
    param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", 1, cfg::n_hidden, 0, 0.01);
    param_dict["w_event2h"] = new LinearParam<mode, Dtype>("w_event2h", cfg::n_embed, cfg::n_hidden, 0, 0.01);
    param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, 0.01);
    param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", cfg::n_hidden, 1, 0, 0.01);
    param_dict["w_event_out"] = new LinearParam<mode, Dtype>("w_event_out", cfg::n_hidden, train_data->num_events, 0, 0.01);

    InitNet(net_train, param_dict, cfg::bptt);
    InitNet(net_test, param_dict, 1);


    int max_iter = (long long)cfg::max_epoch * train_data->num_samples / cfg::bptt / cfg::batch_size;
    int init_iter = cfg::iter;
    
    if (init_iter > 0)
    {
        std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
    }
    
    GraphData<mode, Dtype>* g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_train->node_states->DenseDerived().Zeros(cfg::batch_size, cfg::n_hidden);
    GraphData<mode, Dtype>* g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
    
    Dtype mae, rmse, nll; 

    std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label;
    train_feat["last_hidden"] = g_last_hidden_train;
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {        
        train_feat[fmt::sprintf("event_input_%d", i)] = g_event_input[i];
        train_feat[fmt::sprintf("time_input_%d", i)] = g_time_input[i];

        train_label[fmt::sprintf("nll_%d", i)] = g_event_label[i];
        train_label[fmt::sprintf("mse_%d", i)] = g_time_label[i];
        train_label[fmt::sprintf("mae_%d", i)] = g_time_label[i];        
    }

    for (; cfg::iter <= max_iter; ++cfg::iter)
    {
        if (cfg::iter % cfg::test_interval == 0)
        {
            std::cerr << "testing" << std::endl;
            auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived();
            last_hidden_test.Zeros(test_data->batch_size, cfg::n_hidden); 
            test_data->StartNewEpoch();
            mae = rmse = nll = 0;
            while (test_data->NextBatch(g_last_hidden_test, 
                                        g_event_input[0], 
                                        g_time_input[0], 
                                        g_event_label[0], 
                                        g_time_label[0]))
            {
                net_test.ForwardData({{"event_input_0", g_event_input[0]}, 
                                      {"time_input_0", g_time_input[0]}, 
                                      {"last_hidden", g_last_hidden_test}}, TEST); 
                auto loss_map = net_test.ForwardLabel({{"nll_0", g_event_label[0]}, 
                                                       {"mse_0", g_time_label[0]}, 
                                                       {"mae_0", g_time_label[0]}});

                rmse += loss_map["mse_0"];
                mae += loss_map["mae_0"];            
                nll += loss_map["nll_0"];     
            }
            rmse = sqrt(rmse / test_data->num_samples); 
            mae /= test_data->num_samples;
            nll /= test_data->num_samples;
            std::cerr << fmt::sprintf("time mae: %.4f\t time rmse: %.4f\t event nll: %.4f", mae, rmse, nll) << std::endl;
        } 
        
        if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
        {
            std::cerr << fmt::sprintf("saving model for iter = %d", cfg::iter) << std::endl;
            
        }
        
        train_data->NextBpttBatch(cfg::bptt, 
                                  g_last_hidden_train, 
                                  g_event_input, 
                                  g_time_input, 
                                  g_event_label, 
                                  g_time_label);
        
        net_train.ForwardData(train_feat, TRAIN);
        auto loss_map = net_train.ForwardLabel(train_label);

        if (cfg::iter % cfg::report_interval == 0)
        {
            mae = rmse = nll = 0;
            for (unsigned i = 0; i < cfg::bptt; ++i)
            {
                mae += loss_map[fmt::sprintf("mae_%d", i)];
                rmse += loss_map[fmt::sprintf("mse_%d", i)];  
                nll += loss_map[fmt::sprintf("nll_%d", i)];  
            }
            rmse = sqrt(rmse / cfg::bptt / train_data->batch_size);
            mae /= cfg::bptt * train_data->batch_size;
            nll /= cfg::bptt * train_data->batch_size;
            std::cerr << fmt::sprintf("train iter=%d\t time mae: %.4f\t time rmse: %.4f\t event nll: %.4f", cfg::iter, mae, rmse, nll) << std::endl;            
        }               
        net_train.BackPropagation();
        net_train.UpdateParams(cfg::lr, cfg::l2_penalty, cfg::momentum);                                        
    }
}

int main(const int argc, const char** argv)
{
    cfg::LoadParams(argc, argv);    
	GPUHandle::Init(cfg::dev_id);
    
    ReadRawData();
    
    if (cfg::device_type == CPU)
        Work<CPU>();
    else
        Work<GPU>();     
    GPUHandle::Destroy();
    return 0;
}
