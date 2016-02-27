#ifndef JOINT_NET_H
#define JOINT_NET_H

#include "inet.h"

template<MatMode mode, typename Dtype>
class JointNet : public INet<mode, Dtype>
{
public:

	JointNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

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
            if (cfg::loss_type == LossType::EXP)
                this->train_label[fmt::sprintf("expnll_%d", i)] = this->g_time_label[i];
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
        if (cfg::loss_type == LossType::EXP)
            this->test_label["expnll_0"] = this->g_time_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = 0.0, mae = 0.0, nll = 0.0, err_cnt = 0.0, expnll = 0;
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
        expnll /= cfg::bptt * cfg::batch_size;
        std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f\tnll: %.4f\terr_rate: %.4f", cfg::iter, mae, rmse, nll, err_cnt);
        std::cerr << std::endl;
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) 
	{
		Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"], nll = loss_map["nll_0"];
		rmse = sqrt(rmse / dataset->num_samples);
		mae /= dataset->num_samples;
		nll /= dataset->num_samples;
        Dtype err_cnt = loss_map["err_cnt_0"] / dataset->num_samples;
        std::cerr << fmt::sprintf("test_mae: %.6f\ttest_rmse: %.6f\ttest_nll: %.4f\ttest_err_rate: %.4f", mae, rmse, nll, err_cnt);
        std::cerr << std::endl;        
	}

	virtual void InitParamDict() 
	{
		add_diff< LinearParam >(this->model, "w_embed", train_data->num_events, cfg::n_embed, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
		add_diff< LinearParam >(this->model, "w_time2h", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);

        /*
        if (cfg::gru)
        {
            this->param_dict["w_h2update"] = new LinearParam<mode, Dtype>("w_h2update", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_event2update"] = new LinearParam<mode, Dtype>("w_event2update", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_time2update"] = new LinearParam<mode, Dtype>("w_time2update", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_h2reset"] = new LinearParam<mode, Dtype>("w_h2reset", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_event2reset"] = new LinearParam<mode, Dtype>("w_event2reset", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
            this->param_dict["w_time2reset"] = new LinearParam<mode, Dtype>("w_time2reset", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);                        
        }
        */
        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            add_diff< LinearParam >(this->model, "w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }
        add_diff< LinearParam >(this->model, "w_event_out", hidden_size, train_data->num_events, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_time_out", hidden_size, 1, 0, cfg::w_scale);
	}

    virtual ILayer<mode, Dtype>* AddRecur(std::string name, 
                                          NNGraph<mode, Dtype>& gnn,
                                          ILayer<mode, Dtype> *last_hidden_layer, 
                                          ILayer<mode, Dtype>* event_feat, 
                                          ILayer<mode, Dtype>* time_feat, 
                                          IParam<mode, Dtype>* h2h, 
                                          IParam<mode, Dtype>* e2h,
                                          IParam<mode, Dtype>* t2h)
    {
        return cl< ParamLayer >(gnn, {time_feat, event_feat, last_hidden_layer}, {t2h, e2h, h2h}); 
    }

    virtual ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
                                             NNGraph<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IDiffParam<mode, Dtype>* >& param_dict)
    {
        auto* hidden_layer = AddRecur(fmt::sprintf("hidden_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat, time_feat,
                                      param_dict["w_h2h"], param_dict["w_event2h"], param_dict["w_time2h"]);        
        return cl< ReLULayer >(fmt::sprintf("recurrent_hidden_%d", time_step), gnn, {hidden_layer});
    }
/*
    virtual ILayer<mode, Dtype>* AddGRULayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* event_feat, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        // local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* update_linear = AddRecur(fmt::sprintf("update_linear_%d", time_step), 
                                       gnn, last_hidden_layer, event_feat, time_feat, 
                                       param_dict["w_h2update"], param_dict["w_event2update"], param_dict["w_time2update"]); 
        auto* update_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("update_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(update_linear, update_gate);

        // local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* reset_linear = AddRecur(fmt::sprintf("reset_linear_%d", time_step), 
                                      gnn, last_hidden_layer, event_feat, time_feat, 
                                      param_dict["w_h2reset"], param_dict["w_event2update"], param_dict["w_time2reset"]);         
        auto* reset_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("reset_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(reset_linear, reset_gate);

        // local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        auto* gated_hidden = new PairMulLayer<mode, Dtype>(fmt::sprintf("gated_hidden_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(reset_gate, gated_hidden);
        gnn.AddEdge(last_hidden_layer, gated_hidden);


        // local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        auto* hidden_candidate_linear = AddRecur(fmt::sprintf("hidden_candidate_linear%d", time_step), 
                                          gnn, gated_hidden, event_feat, time_feat,
                                          param_dict["w_h2h"], param_dict["w_event2h"], param_dict["w_time2h"]);
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
*/
	virtual ILayer<mode, Dtype>* AddNetBlocks(int time_step, 
											  NNGraph<mode, Dtype>& gnn, 
											  ILayer<mode, Dtype> *last_hidden_layer, 
                                    		  std::map< std::string, IDiffParam<mode, Dtype>* >& param_dict)
	{
        auto* time_input_layer = cl< InputLayer >(fmt::sprintf("time_input_%d", time_step), gnn, {});
        auto* event_input_layer = cl< InputLayer >(fmt::sprintf("event_input_%d", time_step), gnn, {});

    	auto* embed_layer = cl< ParamLayer >(gnn, {event_input_layer}, {param_dict["w_embed"]});
    	auto* relu_embed_layer = cl< ReLULayer >(gnn, {embed_layer});

    	ILayer<mode, Dtype>* recurrent_output = nullptr;
        if (cfg::gru)
        {
            //recurrent_output = AddGRULayer(time_step, gnn, last_hidden_layer, relu_embed_layer, time_input_layer, param_dict);
        } else
            recurrent_output = AddRNNLayer(time_step, gnn, last_hidden_layer, relu_embed_layer, time_input_layer, param_dict);
    	
        auto* top_hidden = recurrent_output;
        if (cfg::n_h2)
        {
            auto* hidden_2 = cl< ParamLayer >(gnn, {recurrent_output}, {param_dict["w_hidden2"]});
            auto* relu_2 = cl< ReLULayer >(gnn, {hidden_2});
            top_hidden = relu_2;
        } 

        auto* event_output_layer = cl< ParamLayer >(fmt::sprintf("event_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_event_out"]}); 

        auto* time_out_layer = cl< ParamLayer >(fmt::sprintf("time_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_time_out"]});

        cl< ClassNLLCriterionLayer >(fmt::sprintf("nll_%d", time_step), gnn, {event_output_layer}, true);

        cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), 
                                gnn, {time_out_layer}, 
                                cfg::lambda, 
                                cfg::loss_type == LossType::MSE ? PropErr::T : PropErr::N);

        cl< ABSCriterionLayer >(fmt::sprintf("mae_%d", time_step), gnn, {time_out_layer}, PropErr::N);

        cl< ErrCntCriterionLayer >(fmt::sprintf("err_cnt_%d", time_step), gnn, {event_output_layer});

		return recurrent_output; 
	}

    virtual void WriteTestBatch(FILE* fid) override
    {
        /*
        this->net_test.GetDenseNodeState("time_out_0", time_pred);
        this->net_test.GetDenseNodeState("event_out_0", event_pred);
        for (size_t i = 0; i < time_pred.rows; ++i)
        {
            fprintf(fid, "%.6f ", time_pred.data[i]);
            int pred = 0; 
            Dtype best = event_pred.data[i * event_pred.cols];
            for (size_t j = 1; j < event_pred.cols; ++j)
                if (event_pred.data[i * event_pred.cols + j] > best)
                {
                    best = event_pred.data[i * event_pred.cols + j]; 
                    pred = j;
                }
            fprintf(fid, "%d\n", pred);
        }*/
    }

    DenseMat<CPU, Dtype> time_pred, event_pred;
};

#endif