#ifndef EVENT_NET_H
#define EVENT_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class EventNet : public INet<mode, Dtype>
{
public:
	EventNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

	virtual void LinkTrainData() override
	{
    	this->train_feat["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
    		this->train_feat[fmt::sprintf("event_input_%d", i)] = this->g_event_input[i];
        	this->train_label[fmt::sprintf("nll_%d", i)] = this->g_event_label[i];
        	this->train_label[fmt::sprintf("err_cnt_%d", i)] = this->g_event_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_feat["last_hidden"] = this->g_last_hidden_test;
		this->test_feat["event_input_0"] = this->g_event_input[0];
		this->test_label["nll_0"] = this->g_event_label[0];
		this->test_label["err_cnt_0"] = this->g_event_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) override
	{
		Dtype nll = 0.0, err_cnt = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
        	nll += loss_map[fmt::sprintf("nll_%d", i)]; 
        	err_cnt += loss_map[fmt::sprintf("err_cnt_%d", i)]; 
        }
        nll /= cfg::bptt * train_data->batch_size;
        err_cnt /= cfg::bptt * train_data->batch_size;
        std::cerr << fmt::sprintf("train iter=%d\tevent nll: %.4f\tevent err_rate: %.4f", cfg::iter, nll, err_cnt) << std::endl; 
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) override
	{
		Dtype nll = loss_map["nll_0"] / dataset->num_samples;
		Dtype err_cnt = loss_map["err_cnt_0"] / dataset->num_samples;
        std::cerr << fmt::sprintf("event nll: %.4f\tevent err_rate: %.4f", nll, err_cnt) << std::endl;
	}

	virtual void InitParamDict() override
	{
		this->param_dict["w_embed"] = new LinearParam<mode, Dtype>("w_embed",  train_data->num_events, cfg::n_embed, 0, cfg::w_scale);
    	this->param_dict["w_event2h"] = new LinearParam<mode, Dtype>("w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_event_out"] = new LinearParam<mode, Dtype>("w_event_out", cfg::n_hidden, train_data->num_events, 0, cfg::w_scale);
    	this->param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
	}

	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  GraphNN<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict) override
	{
    	gnn.AddLayer(last_hidden_layer);

    	auto* event_input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("event_input_%d", time_step), GraphAtt::NODE);

    	auto* embed_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("embed_%d", time_step), param_dict["w_embed"], GraphAtt::NODE); 

    	auto* relu_embed_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_embed_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);

    	auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    	hidden_layer->AddParam(relu_embed_layer->name, param_dict["w_event2h"], GraphAtt::NODE); 
    	hidden_layer->AddParam(last_hidden_layer->name, param_dict["w_h2h"], GraphAtt::NODE); 

    	auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_hidden_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
    	auto* event_output_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("event_out_%d", time_step), param_dict["w_event_out"], GraphAtt::NODE); 

    	auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>(fmt::sprintf("nll_%d", time_step), true);
    	auto* err_cnt = new ErrCntCriterionLayer<mode, Dtype>(fmt::sprintf("err_cnt_%d", time_step));

    	gnn.AddEdge(event_input_layer, embed_layer);
    	gnn.AddEdge(embed_layer, relu_embed_layer);
    	
    	gnn.AddEdge(relu_embed_layer, hidden_layer);
    	gnn.AddEdge(last_hidden_layer, hidden_layer);
    
    	gnn.AddEdge(hidden_layer, relu_hidden_layer);
    	
    	gnn.AddEdge(relu_hidden_layer, event_output_layer);
       	
    	gnn.AddEdge(event_output_layer, classnll);
    	gnn.AddEdge(event_output_layer, err_cnt); 

		return relu_hidden_layer; 
	}

	virtual void WriteTestBatch(FILE* fid) override
	{        
        this->net_test.GetDenseNodeState("event_out_0", buf);
        for (size_t i = 0; i < buf.rows; ++i)
        {
            int pred = 0; 
            Dtype best = buf.data[i * buf.cols];
            for (size_t j = 1; j < buf.cols; ++j)
                if (buf.data[i * buf.cols + j] > best)
                {
                    best = buf.data[i * buf.cols + j]; 
                    pred = j;
                }
            fprintf(fid, "%d\n", pred);
        }
	}

    DenseMat<CPU, Dtype> buf;
};

#endif
