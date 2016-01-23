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
#include "neural_pointprocess.h"

template<MatMode mode>
ILayer<mode, Dtype>* AddNetBlocks(int time_step, GraphNN<mode, Dtype>& gnn, ILayer<mode, Dtype> *last_hidden_layer, 
                                    std::map< std::string, LinearParam<mode, Dtype>* >& param_dict)
{
    gnn.AddLayer(last_hidden_layer);
    auto* time_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step));

    auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    hidden_layer->AddParam(time_input_layer->name, param_dict["w_time2h"]);
    hidden_layer->AddParam(last_hidden_layer->name, param_dict["w_h2h"]);

    auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_hidden_%d", time_step), WriteType::INPLACE, ActTarget::NODE);

    auto* time_out_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"]); 
    //auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);
    
    auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step));
    auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N);

    typedef typename std::map< std::string, LinearParam<mode, Dtype>* >::iterator param_iter;
    for (param_iter it = param_dict.begin(); it != param_dict.end(); ++it)
        gnn.AddParam(it->second);

    gnn.AddLayer(time_input_layer);
    gnn.AddLayer(hidden_layer);
    gnn.AddLayer(relu_hidden_layer);
    gnn.AddLayer(time_out_layer);
    //gnn.AddLayer(exp_layer);
    
    gnn.AddLayer(mse_criterion);
    gnn.AddLayer(mae_criterion);

    gnn.AddEdge(time_input_layer, hidden_layer);
    gnn.AddEdge(last_hidden_layer, hidden_layer);
    gnn.AddEdge(hidden_layer, relu_hidden_layer);
    gnn.AddEdge(relu_hidden_layer, time_out_layer);
    //gnn.AddEdge(time_out_layer, exp_layer);

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
    InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label); 
    

    GraphNN<mode, Dtype> net_train, net_test;

    std::map< std::string, LinearParam<mode, Dtype>* > param_dict; 
    param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", 1, cfg::n_hidden, 0, 0.001);
    param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, 0.001);
    param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", cfg::n_hidden, 1, 0, 0.001);

    InitNet(net_train, param_dict, cfg::bptt);
    InitNet(net_test, param_dict, 1);

    int max_iter = (long long)cfg::max_epoch * train_data->num_samples / cfg::bptt / cfg::batch_size;
    int init_iter = cfg::iter;
    
    if (init_iter > 0)
    {
        std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
    }
    
    GraphData<mode, Dtype>* g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
    auto& last_hidden_train = g_last_hidden_train->node_states->DenseDerived();     
    GraphData<mode, Dtype>* g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
    
    Dtype mae, rmse; 

    std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label;
    train_feat["last_hidden"] = g_last_hidden_train;
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {        
        train_feat[fmt::sprintf("time_input_%d", i)] = g_time_input[i];
        train_label[fmt::sprintf("mse_%d", i)] = g_time_label[i];
        train_label[fmt::sprintf("mae_%d", i)] = g_time_label[i];        
    }

    last_hidden_train.Zeros(cfg::batch_size, cfg::n_hidden);
    for (; cfg::iter <= max_iter; ++cfg::iter)
    {
        if (cfg::iter % cfg::test_interval == 0)
        {
            std::cerr << "testing" << std::endl;
            auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived();
            last_hidden_test.Zeros(test_data->batch_size, cfg::n_hidden); 
            test_data->StartNewEpoch();
            mae = rmse = 0;
            while (test_data->NextBatch(g_last_hidden_test, 
                                        g_event_input[0], 
                                        g_time_input[0], 
                                        g_event_label[0], 
                                        g_time_label[0]))
            {
                net_test.ForwardData({{"time_input_0", g_time_input[0]}, 
                                      {"last_hidden", g_last_hidden_test}}, TEST); 
                auto loss_map = net_test.ForwardLabel({{"mse_0", g_time_label[0]}, 
                                                       {"mae_0", g_time_label[0]}});

                rmse += loss_map["mse_0"];
                mae += loss_map["mae_0"];
                net_test.GetDenseNodeState("relu_hidden_0", last_hidden_test);
            }
            rmse = sqrt(rmse / test_data->num_samples); 
            mae /= test_data->num_samples;
            std::cerr << fmt::sprintf("time mae: %.4f\t time rmse: %.4f", mae, rmse) << std::endl;
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
        //net_train.GetDenseNodeState(fmt::sprintf("relu_hidden_%d", cfg::bptt - 1), last_hidden_train);

        if (cfg::iter % cfg::report_interval == 0)
        {
            mae = rmse = 0;
            for (unsigned i = 0; i < cfg::bptt; ++i)
            {
                mae += loss_map[fmt::sprintf("mae_%d", i)];
                rmse += loss_map[fmt::sprintf("mse_%d", i)];  
            }
            rmse = sqrt(rmse / cfg::bptt / train_data->batch_size);
            mae /= cfg::bptt * train_data->batch_size;
            std::cerr << fmt::sprintf("train iter=%d\t time mae: %.4f\t time rmse: %.4f", cfg::iter, mae, rmse) << std::endl;            
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
