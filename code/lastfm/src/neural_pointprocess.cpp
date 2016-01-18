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

#include "config.h"
#include "data_loader.h"


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
    
    int max_iter = (long long)cfg::max_epoch * train_data->num_samples / cfg::bptt / cfg::batch_size;
    int init_iter = cfg::iter;
    
    if (init_iter > 0)
    {
        std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
    }
    
    GraphData<mode, Dtype>* g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_train->node_states->DenseDerived().Zeros(cfg::batch_size, cfg::n_hidden);
    GraphData<mode, Dtype>* g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
    
    for (; cfg::iter <= max_iter; ++cfg::iter)
    {
        if (cfg::iter % cfg::test_interval == 0)
        {
            std::cerr << "testing" << std::endl;
            auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived();
            last_hidden_test.Zeros(test_data->batch_size, cfg::n_hidden);
            test_data->StartNewEpoch();
            
            int t = 0;
            while (test_data->NextBatch(g_last_hidden_test, 
                                        g_event_input[0], 
                                        g_time_input[0], 
                                        g_event_label[0], 
                                        g_time_label[0]))
            {
                t++;
            }
            std::cerr << t << std::endl;
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
                                  
        if (cfg::iter % cfg::report_interval == 0)
        {
            
        }                                              
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
