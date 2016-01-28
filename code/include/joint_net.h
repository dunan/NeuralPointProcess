#ifndef JOINT_NET_H
#define JOINT_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class JointNet : public INet<mode, Dtype>
{
public:

	JointNet() : INet<mode, Dtype>() {}

	virtual void LinkTrainData() override 
	{
    	this->train_feat["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
    		this->train_feat[fmt::sprintf("event_input_%d", i)] = this->g_event_input[i];
        	this->train_label[fmt::sprintf("nll_%d", i)] = this->g_event_label[i];
        	this->train_feat[fmt::sprintf("time_input_%d", i)] = this->g_time_input[i];
        	this->train_label[fmt::sprintf("mse_%d", i)] = this->g_time_label[i];
        	this->train_label[fmt::sprintf("mae_%d", i)] = this->g_time_label[i];
            this->train_label[fmt::sprintf("err_cnt_%d", i)] = this->g_event_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_feat["last_hidden"] = this->g_last_hidden_test;
		this->test_feat["event_input_0"] = this->g_event_input[0];
		this->test_feat["time_input_0"] = this->g_time_input[0];
		this->test_label["mse_0"] = this->g_time_label[0];
		this->test_label["mae_0"] = this->g_time_label[0];
		this->test_label["nll_0"] = this->g_event_label[0];
        this->test_label["err_cnt_0"] = this->g_event_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = 0.0, mae = 0.0, nll = 0.0, err_cnt = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
            rmse += loss_map[fmt::sprintf("mse_%d", i)];
        	nll += loss_map[fmt::sprintf("nll_%d", i)]; 
            err_cnt += loss_map[fmt::sprintf("err_cnt_%d", i)]; 
        }
        rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
		mae /= cfg::bptt * cfg::batch_size;
		nll /= cfg::bptt * train_data->batch_size;
        err_cnt /= cfg::bptt * train_data->batch_size;
        std::cerr << fmt::sprintf("train iter=%d\ttime mae: %.4f\t time rmse: %.4f\t event nll: %.4f\tevent err_rate: %.4f", cfg::iter, mae, rmse, nll, err_cnt) << std::endl; 
	}

	virtual void PrintTestResults(std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"], nll = loss_map["nll_0"];
		rmse = sqrt(rmse / test_data->num_samples);
		mae /= test_data->num_samples;
		nll /= test_data->num_samples;
        Dtype err_cnt = loss_map["err_cnt_0"] / test_data->num_samples;
        std::cerr << fmt::sprintf("time mae: %.4f\t time rmse: %.4f\t event nll: %.4f\tevent err_rate: %.4f", mae, rmse, nll, err_cnt) << std::endl;
	}

	virtual void InitParamDict() 
	{
		this->param_dict["w_embed"] = new LinearParam<mode, Dtype>("w_embed",  train_data->num_events, cfg::n_embed, 0, cfg::w_scale);
    	this->param_dict["w_event2h"] = new LinearParam<mode, Dtype>("w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_event_out"] = new LinearParam<mode, Dtype>("w_event_out", cfg::n_hidden, train_data->num_events, 0, cfg::w_scale);
		this->param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", 1, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", cfg::n_hidden, 1, 0, cfg::w_scale);
	}

	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  GraphNN<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, LinearParam<mode, Dtype>* >& param_dict)
	{
    	gnn.AddLayer(last_hidden_layer);
    	auto* event_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("event_input_%d", time_step), GraphAtt::NODE);
    	auto* time_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step), GraphAtt::NODE);

    	auto* embed_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("embed_%d", time_step), param_dict["w_embed"], GraphAtt::NODE); 

    	auto* relu_embed_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_embed_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);

    	auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    	hidden_layer->AddParam(time_input_layer->name, param_dict["w_time2h"], GraphAtt::NODE); 
    	hidden_layer->AddParam(relu_embed_layer->name, param_dict["w_event2h"], GraphAtt::NODE); 
    	hidden_layer->AddParam(last_hidden_layer->name, param_dict["w_h2h"], GraphAtt::NODE); 

    	auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_hidden_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
    	auto* event_output_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("event_out_%d", time_step), param_dict["w_event_out"], GraphAtt::NODE); 

    	auto* time_out_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"], GraphAtt::NODE); 
    	auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);

    	auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>(fmt::sprintf("nll_%d", time_step), true);
    	auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step));
    	auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N);
        auto* err_cnt = new ErrCntCriterionLayer<mode, Dtype>(fmt::sprintf("err_cnt_%d", time_step));

    	gnn.AddEdge(event_input_layer, embed_layer);
    	gnn.AddEdge(embed_layer, relu_embed_layer);
    	
    	gnn.AddEdge(time_input_layer, hidden_layer);
    	gnn.AddEdge(relu_embed_layer, hidden_layer);
    	gnn.AddEdge(last_hidden_layer, hidden_layer);
    
    	gnn.AddEdge(hidden_layer, relu_hidden_layer);
    	
    	gnn.AddEdge(relu_hidden_layer, event_output_layer);
    	gnn.AddEdge(relu_hidden_layer, time_out_layer);
    
    	gnn.AddEdge(time_out_layer, exp_layer);
    	gnn.AddEdge(exp_layer, mse_criterion);
    	gnn.AddEdge(exp_layer, mae_criterion);
    	
    	gnn.AddEdge(event_output_layer, classnll);
        gnn.AddEdge(event_output_layer, err_cnt); 

		return relu_hidden_layer; 
	}

    virtual void WriteTestBatch(FILE* fid) override
    {
        
    }
};

#endif