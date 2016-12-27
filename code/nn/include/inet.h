#ifndef INET_H
#define INET_H

#include "dense_matrix.h"
#include "linear_param.h"
#include "const_scalar_param.h"
#include "nngraph.h"
#include "param_layer.h"
#include "input_layer.h"
#include "c_add_layer.h"
#include "c_mul_layer.h"
#include "fmt/format.h"
#include "relu_layer.h"
#include "model.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "sigmoid_layer.h"
#include "classnll_criterion_layer.h"
#include "config.h"
#include "data_loader.h"
#include "err_cnt_criterion_layer.h"
#include "intensity_nll_criterion_layer.h"
#include "dur_pred_layer.h"
#include "learner.h"

template<MatMode mode, typename Dtype>
class INet
{
public:
	INet(IEventTimeLoader<mode>* _etloader)
	{
        this->etloader = _etloader;
        initialized = false;
        g_last_hidden_train = new DenseMat<mode, Dtype>();
        g_last_hidden_test = new DenseMat<mode, Dtype>();
		InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label);
        learner = new MomentumSGDLearner<mode, Dtype>(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
	}

    void Setup()
    {
        InitParamDict();

        InitNet(net_train, model.all_params, cfg::bptt);
        InitNet(net_test, model.all_params, 1);
        initialized = true;
    }

    void EvaluateDataset(const char* prefix, DataLoader<TEST>* dataset, bool save_prediction, std::map<std::string, Dtype>& test_loss_map)
    {
        auto& last_hidden_test = g_last_hidden_test->DenseDerived();
        last_hidden_test.Zeros(dataset->batch_size, cfg::n_hidden); 

        dataset->StartNewEpoch();
                
        test_loss_map.clear();
        FILE* fid = nullptr;
        if (save_prediction)
            fid = fopen(fmt::sprintf("%s/%s_pred_iter_%d.txt", cfg::save_dir, prefix, cfg::iter).c_str(), "w");
        
        while (dataset->NextBatch(etloader, 
                                  g_last_hidden_test, 
                                  g_event_input[0], 
                                  g_time_input[0], 
                                  g_event_label[0], 
                                  g_time_label[0]))
        {                        
            net_test.FeedForward(test_dict, TEST);
            auto loss_map = net_test.GetLoss();

            for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
            {
                if (test_loss_map.count(it->first) == 0)
                    test_loss_map[it->first] = 0.0;
                test_loss_map[it->first] += it->second;
            }
            if (save_prediction)
                WriteTestBatch(fid);
            if (cfg::bptt > 1)
                net_test.GetState("recurrent_hidden_0", last_hidden_test);            
        }
        if (save_prediction)
            fclose(fid);
    }

	void MainLoop()
	{
        if (!initialized)
            Setup();

		long long max_iter = (long long)cfg::max_epoch;
    	int init_iter = cfg::iter;
    
    	if (init_iter > 0)
    	{
        	std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
            model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
    	}
        			
    	LinkTrainData();
    	LinkTestData();

        auto& last_hidden_train = g_last_hidden_train->DenseDerived();         
    	last_hidden_train.Zeros(cfg::batch_size, cfg::n_hidden);
    	std::map<std::string, Dtype> test_loss_map;

    	for (; cfg::iter <= max_iter; ++cfg::iter)
    	{
        	if (cfg::iter % cfg::test_interval == 0)
	        {
    	        std::cerr << "testing" << std::endl;
        	    
                EvaluateDataset("test", test_data, true, test_loss_map);
                PrintTestResults(test_data, test_loss_map);
                if (cfg::has_eval)
                {
                    EvaluateDataset("val", val_data, cfg::save_eval, test_loss_map);
                    PrintTestResults(val_data, test_loss_map);    
                }
        	}
        
        	if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
        	{
            	std::cerr << fmt::sprintf("saving model for iter = %d", cfg::iter) << std::endl;
                model.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
        	}
        
        	train_data->NextBpttBatch(etloader, 
                                      cfg::bptt, 
            	                      g_last_hidden_train, 
                	                  g_event_input, 
                    	              g_time_input, 
                        	          g_event_label, 
                            	      g_time_label);
        
        	net_train.FeedForward(train_dict, TRAIN);
        	auto loss_map = net_train.GetLoss();
            if (cfg::bptt > 1 && cfg::use_history)
            {
                net_train.GetState(fmt::sprintf("recurrent_hidden_%d", cfg::bptt - 1), last_hidden_train);
            }

            net_train.BackPropagation();
            learner->Update();   

        	if (cfg::iter % cfg::report_interval == 0)
        	{
        		PrintTrainBatchResults(loss_map);
        	}
    	}
	}

	NNGraph<mode, Dtype> net_train, net_test;
    Model<mode, Dtype> model;
    MomentumSGDLearner<mode, Dtype>* learner;

	std::vector< IMatrix<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;	
	std::map<std::string, IMatrix<mode, Dtype>* > train_dict, test_dict;
    IEventTimeLoader<mode>* etloader;

    bool initialized;
    void InitNet(NNGraph<mode, Dtype>& gnn, 
                 std::map< std::string, IParam<mode, Dtype>* >& param_dict, 
                 unsigned n_unfold)
    {
        auto* last_hidden_layer = cl< InputLayer >("last_hidden", gnn, {});

        for (unsigned i = 0; i < n_unfold; ++i)
        {
            auto* new_hidden = AddNetBlocks(i, gnn, last_hidden_layer, param_dict);
            last_hidden_layer = new_hidden;
        }
    }    
    IMatrix<mode, Dtype>* g_last_hidden_train, *g_last_hidden_test;

    virtual void WriteTestBatch(FILE* fid) = 0;
	virtual void LinkTrainData() = 0;
	virtual void LinkTestData() = 0;
	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) = 0;
	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) = 0;

	virtual void InitParamDict() = 0;
	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  NNGraph<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict) = 0;
};

#endif
