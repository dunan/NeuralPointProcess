#ifndef INET_H
#define INET_H

#include "dense_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "param_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "exp_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "classnll_criterion_layer.h"
#include "config.h"
#include "data_loader.h"
#include "err_cnt_criterion_layer.h"

template<MatMode mode, typename Dtype>
class INet
{
public:
	INet()
	{
        initialized = false;
        g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
        g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
		InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label); 
	}

    void Setup()
    {
        InitParamDict();

        InitNet(net_train, param_dict, cfg::bptt);
        InitNet(net_test, param_dict, 1);
        initialized = true;
    }

    void EvaluateDataset(DataLoader<TEST>* dataset, bool save_prediction, std::map<std::string, Dtype>& test_loss_map)
    {
        auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived();
        last_hidden_test.Zeros(dataset->batch_size, cfg::n_hidden); 

        dataset->StartNewEpoch();
                
        test_loss_map.clear();
        FILE* fid = nullptr;
        if (save_prediction)
            fid = fopen(fmt::sprintf("%s/pred_iter_%d.txt", cfg::save_dir, cfg::iter).c_str(), "w");

        while (dataset->NextBatch(g_last_hidden_test, 
                                  g_event_input[0], 
                                  g_time_input[0], 
                                  g_event_label[0], 
                                  g_time_label[0]))
        {
            net_test.ForwardData(test_feat, TEST);
            auto loss_map = net_test.ForwardLabel(test_label);

            for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
            {
                if (test_loss_map.count(it->first) == 0)
                    test_loss_map[it->first] = 0.0;
                test_loss_map[it->first] += it->second;
            }
            if (save_prediction)
                WriteTestBatch(fid);
            if (cfg::bptt > 1)
                net_test.GetDenseNodeState("relu_hidden_0", last_hidden_test);
        }
        if (save_prediction)
            fclose(fid);
    }

	void MainLoop()
	{
        if (!initialized)
            Setup();

		int max_iter = (long long)cfg::max_epoch * train_data->num_samples / cfg::bptt / cfg::batch_size;
    	int init_iter = cfg::iter;
    
    	if (init_iter > 0)
    	{
        	std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
            net_train.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
    	}
        			
    	LinkTrainData();
    	LinkTestData();

        auto& last_hidden_train = g_last_hidden_train->node_states->DenseDerived();         
    	last_hidden_train.Zeros(cfg::batch_size, cfg::n_hidden);
    	std::map<std::string, Dtype> test_loss_map;

    	for (; cfg::iter <= max_iter; ++cfg::iter)
    	{
        	if (cfg::iter % cfg::test_interval == 0)
	        {
    	        std::cerr << "testing" << std::endl;
        	    
                EvaluateDataset(test_data, false, test_loss_map);
                PrintTestResults(test_data, test_loss_map);
                if (cfg::has_eval)
                {
                    EvaluateDataset(val_data, cfg::save_eval, test_loss_map);
                    PrintTestResults(val_data, test_loss_map);    
                }                
        	}
        
        	if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
        	{
            	std::cerr << fmt::sprintf("saving model for iter = %d", cfg::iter) << std::endl;
                net_train.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
        	}
        
        	train_data->NextBpttBatch(cfg::bptt, 
            	                      g_last_hidden_train, 
                	                  g_event_input, 
                    	              g_time_input, 
                        	          g_event_label, 
                            	      g_time_label);
        
        	net_train.ForwardData(train_feat, TRAIN);
        	auto loss_map = net_train.ForwardLabel(train_label);
            if (cfg::bptt > 1)
            {
                net_train.GetDenseNodeState(fmt::sprintf("relu_hidden_%d", cfg::bptt - 1), last_hidden_train);
            }

            net_train.BackPropagation();
            net_train.UpdateParams(cfg::lr, cfg::l2_penalty, cfg::momentum);   

        	if (cfg::iter % cfg::report_interval == 0)
        	{
        		PrintTrainBatchResults(loss_map);
        	}
    	}
	}


	std::map< std::string, LinearParam<mode, Dtype>* > param_dict;
	GraphNN<mode, Dtype> net_train, net_test;
	std::vector< GraphData<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;	
	std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label, test_feat, test_label;

    bool initialized;
    void InitNet(GraphNN<mode, Dtype>& gnn, 
                 std::map< std::string, LinearParam<mode, Dtype>* >& param_dict, 
                 unsigned n_unfold)
    {
        ILayer<mode, Dtype>* last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden", GraphAtt::NODE);

        for (auto it = param_dict.begin(); it != param_dict.end(); ++it)
        {
            gnn.AddParam(it->second);
        }

        for (unsigned i = 0; i < n_unfold; ++i)
        {
            auto* new_hidden = AddNetBlocks(i, gnn, last_hidden_layer, param_dict);
            last_hidden_layer = new_hidden;
        }
    }    

    GraphData<mode, Dtype>* g_last_hidden_train, *g_last_hidden_test;

    virtual void WriteTestBatch(FILE* fid) = 0;
	virtual void LinkTrainData() = 0;
	virtual void LinkTestData() = 0;
	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) = 0;
	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) = 0;

	virtual void InitParamDict() = 0;
	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  GraphNN<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, LinearParam<mode, Dtype>* >& param_dict) = 0;
};

#endif