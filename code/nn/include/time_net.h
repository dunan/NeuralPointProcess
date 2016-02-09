#ifndef TIME_NET_H
#define TIME_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class TimeNet : public INet<mode, Dtype>
{
public:
	TimeNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

	virtual void LinkTrainData() override
	{
    	this->train_feat["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
        	this->train_feat[fmt::sprintf("time_input_%d", i)] = this->g_time_input[i];
        	this->train_label[fmt::sprintf("mse_%d", i)] = this->g_time_label[i];
        	this->train_label[fmt::sprintf("mae_%d", i)] = this->g_time_label[i];
        	if (cfg::loss_type == LossType::EXP)
        		this->train_label[fmt::sprintf("expnll_%d", i)] = this->g_time_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_feat["last_hidden"] = this->g_last_hidden_test;
		this->test_feat["time_input_0"] = this->g_time_input[0];
		this->test_label["mse_0"] = this->g_time_label[0];
		this->test_label["mae_0"] = this->g_time_label[0];
		if (cfg::loss_type == LossType::EXP)
			this->test_label["expnll_0"] = this->g_time_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) override
	{
		Dtype rmse = 0.0, mae = 0.0, expnll = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
            rmse += loss_map[fmt::sprintf("mse_%d", i)];  
            if (cfg::loss_type == LossType::EXP)
            	expnll += loss_map[fmt::sprintf("expnll_%d", i)];  
        }
        rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
		mae /= cfg::bptt * cfg::batch_size;
		expnll /= cfg::bptt * cfg::batch_size;
		std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f", cfg::iter, mae, rmse);
		if (cfg::loss_type == LossType::EXP)
			std::cerr << fmt::sprintf("\texpnll: %.4f", expnll);
		std::cerr << std::endl;
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) override
	{
		Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"];
		rmse = sqrt(rmse / dataset->num_samples);
		mae /= dataset->num_samples;
		std::cerr << fmt::sprintf("test mae: %.4f\t test rmse: %.4f", mae, rmse);
		if (cfg::loss_type == LossType::EXP)
			std::cerr << fmt::sprintf("\texpnll: %.4f", loss_map["expnll_0"] / dataset->num_samples);
		std::cerr << std::endl;
	}

	virtual void InitParamDict() override
	{
		this->param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", 1, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", cfg::n_hidden, 1, 0, cfg::w_scale);
	}

	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  GraphNN<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, LinearParam<mode, Dtype>* >& param_dict) override
	{
		gnn.AddLayer(last_hidden_layer);
    
		auto* time_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step), GraphAtt::NODE);

    	auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    	hidden_layer->AddParam(time_input_layer->name, param_dict["w_time2h"], GraphAtt::NODE);
    	hidden_layer->AddParam(last_hidden_layer->name, param_dict["w_h2h"], GraphAtt::NODE);

    	auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_hidden_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);

	    auto* time_out_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"], GraphAtt::NODE); 
	    //auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
    
    	auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step),  
    															 cfg::loss_type == LossType::MSE ? PropErr::T : PropErr::N);
    	auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step),  PropErr::N);

    	gnn.AddEdge(time_input_layer, hidden_layer);
    	gnn.AddEdge(last_hidden_layer, hidden_layer);
    	gnn.AddEdge(hidden_layer, relu_hidden_layer);
    	gnn.AddEdge(relu_hidden_layer, time_out_layer);
    	//gnn.AddEdge(time_out_layer, exp_layer);

    	gnn.AddEdge(time_out_layer, mse_criterion);
    	gnn.AddEdge(time_out_layer, mae_criterion);

    	if (cfg::loss_type == LossType::EXP)
    	{
    		auto* expnll_criterion = new ExpNLLCriterionLayer<mode, Dtype>(fmt::sprintf("expnll_%d", time_step));
    		gnn.AddEdge(time_out_layer, expnll_criterion); 
    	}

    	return relu_hidden_layer; 
	}

	virtual void WriteTestBatch(FILE* fid) override
	{
		this->net_test.GetDenseNodeState("time_out_0", buf);		
		buf2.CopyFrom(this->g_time_label[0]->node_states->DenseDerived());
        for (size_t i = 0; i < buf.rows; ++i)
            fprintf(fid, "%.6f %.6f\n",  buf.data[i], buf2.data[i]);
	}

	DenseMat<CPU, Dtype> buf, buf2;
};

#endif