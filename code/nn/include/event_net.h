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
        std::cerr << fmt::sprintf("train iter=%d\tnll: %.4f\terr_rate: %.4f", cfg::iter, nll, err_cnt) << std::endl; 
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) override
	{
		Dtype nll = loss_map["nll_0"] / dataset->num_samples;
		Dtype err_cnt = loss_map["err_cnt_0"] / dataset->num_samples;
        std::cerr << fmt::sprintf("test_nll: %.4f\ttest_err_rate: %.4f", nll, err_cnt) << std::endl;
	}

	virtual void InitParamDict() override
	{
		this->param_dict["w_embed"] = new LinearParam<mode, Dtype>("w_embed",  train_data->num_events, cfg::n_embed, 0, cfg::w_scale);
    	this->param_dict["w_event2h"] = new LinearParam<mode, Dtype>("w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
    	this->param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);

        if (cfg::gru)
        {
            this->param_dict["w_h2update"] = new LinearParam<mode, Dtype>("w_h2update", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_event2update"] = new LinearParam<mode, Dtype>("w_event2update", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_h2reset"] = new LinearParam<mode, Dtype>("w_h2reset", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_event2reset"] = new LinearParam<mode, Dtype>("w_event2reset", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
        }
        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            this->param_dict["w_hidden2"] = new LinearParam<mode, Dtype>("w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }

        this->param_dict["w_event_out"] = new LinearParam<mode, Dtype>("w_event_out", hidden_size, train_data->num_events, 0, cfg::w_scale);
	}

    virtual ILayer<mode, Dtype>* AddRecur(std::string name, 
                                          GraphNN<mode, Dtype>& gnn,
                                          ILayer<mode, Dtype> *last_hidden_layer, 
                                          ILayer<mode, Dtype>* event_feat, 
                                          IParam<mode, Dtype>* h2h, 
                                          IParam<mode, Dtype>* e2h)
    {
        auto* hidden_layer = new NodeLayer<mode, Dtype>(name); 
        hidden_layer->AddParam(event_feat->name, e2h, GraphAtt::NODE); 
        hidden_layer->AddParam(last_hidden_layer->name, h2h, GraphAtt::NODE); 

        gnn.AddEdge(event_feat, hidden_layer);
        gnn.AddEdge(last_hidden_layer, hidden_layer);
        return hidden_layer;
    }    

    virtual ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        auto* hidden_layer = AddRecur(fmt::sprintf("hidden_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat,
                                      param_dict["w_h2h"], param_dict["w_event2h"]);

        auto* relu_hidden_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("recurrent_hidden_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(hidden_layer, relu_hidden_layer);

        return relu_hidden_layer;
    }

    virtual ILayer<mode, Dtype>* AddGRULayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        // local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* update_linear = AddRecur(fmt::sprintf("update_linear_%d", time_step), 
                                       gnn, last_hidden_layer, event_feat,
                                       param_dict["w_h2update"], param_dict["w_event2update"]); 
        auto* update_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("update_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(update_linear, update_gate);

        // local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* reset_linear = AddRecur(fmt::sprintf("reset_linear_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat, 
                                      param_dict["w_h2reset"], param_dict["w_event2update"]);         
        auto* reset_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("reset_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(reset_linear, reset_gate);

        // local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        auto* gated_hidden = new PairMulLayer<mode, Dtype>(fmt::sprintf("gated_hidden_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(reset_gate, gated_hidden);
        gnn.AddEdge(last_hidden_layer, gated_hidden);


        // local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        auto* hidden_candidate_linear = AddRecur(fmt::sprintf("hidden_candidate_linear%d", time_step), 
                                          gnn, gated_hidden, event_feat,
                                          param_dict["w_h2h"], param_dict["w_event2h"]);
        auto* hidden_candidate = new ReLULayer<mode, Dtype>(fmt::sprintf("hidden_candidate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(hidden_candidate_linear, hidden_candidate);

        // local zh = nn.CMulTable()({update_gate, hidden_candidate})
        auto* zh = new PairMulLayer<mode, Dtype>(fmt::sprintf("zh_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(hidden_candidate, zh);
        gnn.AddEdge(update_gate, zh);

        // nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate))
        auto* z_prev_h = new ConstTransLayer<mode, Dtype>(fmt::sprintf("z_prev_h_%d", time_step), GraphAtt::NODE, -1, 1);
        gnn.AddEdge(update_gate, z_prev_h);

        // (1 - update_gate) * prev_h
        auto* zhm1 = new PairMulLayer<mode, Dtype>(fmt::sprintf("zhm1_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(z_prev_h, zhm1);
        gnn.AddEdge(last_hidden_layer, zhm1);

        // local next_h = nn.CAddTable()({zh, zhm1})
        auto* next_h = new NodeGatherLayer<mode, Dtype>(fmt::sprintf("recurrent_hidden_%d", time_step));
        gnn.AddEdge(zh, next_h);
        gnn.AddEdge(zhm1, next_h);

        return next_h;
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
        gnn.AddEdge(event_input_layer, embed_layer);
        gnn.AddEdge(embed_layer, relu_embed_layer);

        ILayer<mode, Dtype>* recurrent_output = nullptr;
        if (cfg::gru)
        {
            recurrent_output = AddGRULayer(time_step, gnn, last_hidden_layer, relu_embed_layer, param_dict);
        } else 
            recurrent_output = AddRNNLayer(time_step, gnn, last_hidden_layer, relu_embed_layer, param_dict);     

        auto* top_hidden = recurrent_output;
        if (cfg::n_h2)
        {
            auto* hidden_2 = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("hidden_2_%d", time_step), param_dict["w_hidden2"], GraphAtt::NODE);
            gnn.AddEdge(recurrent_output, hidden_2);
            auto* relu_2 = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_h2_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
            gnn.AddEdge(hidden_2, relu_2);
            top_hidden = relu_2;
        }

    	auto* event_output_layer = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("event_out_%d", time_step), param_dict["w_event_out"], GraphAtt::NODE); 
        gnn.AddEdge(top_hidden, event_output_layer);
    	
        auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>(fmt::sprintf("nll_%d", time_step), true);
    	auto* err_cnt = new ErrCntCriterionLayer<mode, Dtype>(fmt::sprintf("err_cnt_%d", time_step));
       	
    	gnn.AddEdge(event_output_layer, classnll);
    	gnn.AddEdge(event_output_layer, err_cnt); 

		return recurrent_output; 
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
