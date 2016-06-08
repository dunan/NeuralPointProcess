#ifndef TIME_NET_H
#define TIME_NET_H

#include "inet.h"
#include "exp_layer.h"

template<MatMode mode, typename Dtype>
class TimeNet : public INet<mode, Dtype>
{
public:
	TimeNet(IEventTimeLoader<mode>* _etloader) : INet<mode, Dtype>(_etloader) {}

	virtual void LinkTrainData() override
	{
    	this->train_dict["last_hidden"] = this->g_last_hidden_train;
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
        	this->train_dict[fmt::sprintf("time_input_%d", i)] = this->g_time_input[i];
        	
            this->train_dict[fmt::sprintf("dur_%d", i)] = this->g_time_label[i];
    	}
	}

	virtual void LinkTestData() override
	{
		this->test_dict["last_hidden"] = this->g_last_hidden_test;
		this->test_dict["time_input_0"] = this->g_time_input[0];
		this->test_dict["dur_0"] = this->g_time_label[0];
	}

	virtual void PrintTrainBatchResults(std::map<std::string, Dtype>& loss_map) override
	{
		Dtype rmse = 0.0, mae = 0.0, intnll = 0.0;
		for (unsigned i = 0; i < cfg::bptt; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
            rmse += loss_map[fmt::sprintf("mse_%d", i)];
            if (cfg::loss_type == LossType::INTENSITY)
                intnll +=  loss_map[fmt::sprintf("intnll_%d", i)]; 
        }
        rmse = sqrt(rmse / cfg::bptt / cfg::batch_size);
		mae /= cfg::bptt * cfg::batch_size;
        intnll /= cfg::bptt * cfg::batch_size;
		std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f", cfg::iter, mae, rmse);
        if (cfg::loss_type == LossType::INTENSITY)
            std::cerr << fmt::sprintf("\tintnll: %.4f", intnll);
		std::cerr << std::endl;
	}

	virtual void PrintTestResults(DataLoader<TEST>* dataset, std::map<std::string, Dtype>& loss_map) override
	{			
        Dtype rmse = loss_map["mse_0"], mae = loss_map["mae_0"];
        rmse = sqrt(rmse / dataset->num_samples);
        mae /= dataset->num_samples;
        std::cerr << fmt::sprintf("test_mae: %.6f\t test_rmse: %.6f", mae, rmse);    

        if (cfg::loss_type == LossType::INTENSITY)
        {
            std::cerr << fmt::sprintf("\ttest_intnll: %.6f", loss_map["intnll_0"] / dataset->num_samples);       
        }
		
		std::cerr << std::endl;
	}

	virtual void InitParamDict() override
	{        
        add_diff< LinearParam >(this->model, "w_time2h", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
    	add_diff< LinearParam >(this->model, "w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);

/*
    	if (cfg::gru)
        {
            this->model.add_diff< LinearParam >("w_h2update", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->model.add_diff< LinearParam >("w_time2update", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
            this->model.add_diff< LinearParam >("w_h2reset", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
            this->model.add_diff< LinearParam >("w_time2reset", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);                        
        }
*/
        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            add_diff< LinearParam >(this->model, "w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }

        add_diff< LinearParam >(this->model, "w_time_out", hidden_size, 1, 0, cfg::w_scale); 
        if (cfg::loss_type == LossType::INTENSITY)
            add_diff< LinearParam >(this->model, "w_lambdat", 1, 1, 0, cfg::w_scale, BiasOption::NONE); 
	}

    ILayer<mode, Dtype>* AddRecur(std::string name, 
                                          NNGraph<mode, Dtype>& gnn,
                                          ILayer<mode, Dtype> *last_hidden_layer, 
                                          ILayer<mode, Dtype>* time_feat, 
                                          IParam<mode, Dtype>* h2h, 
                                          IParam<mode, Dtype>* t2h)
    {
        return cl< ParamLayer >(gnn, {time_feat, last_hidden_layer}, {t2h, h2h}); 
    }

    ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
                                             NNGraph<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        auto* hidden_layer = AddRecur(fmt::sprintf("hidden_%d", time_step), 
                                      gnn, last_hidden_layer, time_feat,
                                      param_dict["w_h2h"], param_dict["w_time2h"]);        

        auto* relu_hidden_layer = cl< ReLULayer >(fmt::sprintf("recurrent_hidden_%d", time_step), gnn, {hidden_layer});

        return relu_hidden_layer;
    }

/*
    virtual ILayer<mode, Dtype>* AddGRULayer(int time_step, 
                                             GraphNN<mode, Dtype>& gnn,
                                             ILayer<mode, Dtype> *last_hidden_layer, 
                                             ILayer<mode, Dtype>* time_feat, 
                                             std::map< std::string, IParam<mode, Dtype>* >& param_dict)
    {
        // local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* update_linear = AddRecur(fmt::sprintf("update_linear_%d", time_step), 
                                       gnn, last_hidden_layer, time_feat, 
                                       param_dict["w_h2update"], param_dict["w_time2update"]); 
        auto* update_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("update_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(update_linear, update_gate);

        // local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
        auto* reset_linear = AddRecur(fmt::sprintf("reset_linear_%d", time_step), 
                                      gnn, last_hidden_layer, time_feat, 
                                      param_dict["w_h2reset"], param_dict["w_time2reset"]);         
        auto* reset_gate = new SigmoidLayer<mode, Dtype>(fmt::sprintf("reset_gate_%d", time_step), GraphAtt::NODE, WriteType::INPLACE);
        gnn.AddEdge(reset_linear, reset_gate);

        // local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
        auto* gated_hidden = new PairMulLayer<mode, Dtype>(fmt::sprintf("gated_hidden_%d", time_step), GraphAtt::NODE);
        gnn.AddEdge(reset_gate, gated_hidden);
        gnn.AddEdge(last_hidden_layer, gated_hidden);

        // local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
        auto* hidden_candidate_linear = AddRecur(fmt::sprintf("hidden_candidate_linear%d", time_step), 
                                          gnn, gated_hidden, time_feat,
                                          param_dict["w_h2h"], param_dict["w_time2h"]);
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
                                    		  std::map< std::string, IParam<mode, Dtype>* >& param_dict) override
	{		    
		auto* time_input_layer = cl< InputLayer >(fmt::sprintf("time_input_%d", time_step), gnn, {});
        auto* dur_label_layer = cl< InputLayer >(fmt::sprintf("dur_%d", time_step), gnn, {});

		ILayer<mode, Dtype>* recurrent_output = nullptr;
        if (cfg::gru)
        {
            //recurrent_output = AddGRULayer(time_step, gnn, last_hidden_layer, time_input_layer, param_dict);
        } else
            recurrent_output = AddRNNLayer(time_step, gnn, last_hidden_layer, time_input_layer, param_dict);

        auto* top_hidden = recurrent_output;
        if (cfg::n_h2)
        {
            auto* hidden_2 = cl< ParamLayer >(gnn, {recurrent_output}, {param_dict["w_hidden2"]});
            auto* relu_2 = cl< ReLULayer >(gnn, {hidden_2});
            top_hidden = relu_2;
        }

	    auto* time_out_layer = cl< ParamLayer >(fmt::sprintf("time_out_%d", time_step), gnn, {top_hidden}, {param_dict["w_time_out"]});
	    
        if (cfg::loss_type == LossType::MSE)
        {
            cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), gnn, {time_out_layer, dur_label_layer});
            cl< ABSCriterionLayer >(fmt::sprintf("mae_%d", time_step), gnn, {time_out_layer, dur_label_layer}, PropErr::N);
        }
        if (cfg::loss_type == LossType::INTENSITY)
        {   
            LinearParam<mode, Dtype>* w = dynamic_cast<LinearParam<mode, Dtype>*>(param_dict["w_lambdat"]);
            cl< IntensityNllCriterionLayer >(fmt::sprintf("intnll_%d", time_step), 
                                             gnn, 
                                             {time_out_layer, dur_label_layer}, 
                                             w);
            if (&gnn == &(this->net_test))
            {
                auto* dur_pred = cl< DurPredLayer >(fmt::sprintf("dur_pred_%d", time_step), 
                                                gnn, 
                                                {time_out_layer}, 
                                                w);    
                cl< MSECriterionLayer >(fmt::sprintf("mse_%d", time_step), gnn, {dur_pred, dur_label_layer}, PropErr::N);
                cl< ABSCriterionLayer >(fmt::sprintf("mae_%d", time_step), gnn, {dur_pred, dur_label_layer}, PropErr::N);
            }
            
            auto* intensity_linear = cl< ParamLayer >(gnn, {top_hidden, dur_label_layer}, {param_dict["w_time_out"], param_dict["w_lambdat"]}, PropErr::N);
            cl< ExpLayer >(fmt::sprintf("intensity_%d", time_step), gnn, {intensity_linear});

            
        }
    	
    	return recurrent_output; 
	}

	virtual void WriteTestBatch(FILE* fid) override
	{           
        this->net_test.GetState("intensity_0", buf);     
        for (size_t i = 0; i < buf.rows; ++i)
            fprintf(fid, "%.6f\n",  buf.data[i]);
        return;
        /*
		this->net_test.GetState("dur_pred_0", buf);
		buf2.CopyFrom(this->g_time_label[0]->DenseDerived());
        for (size_t i = 0; i < buf.rows; ++i)
            fprintf(fid, "%.6f %.6f\n",  buf.data[i], buf2.data[i]);*/
	}

	DenseMat<CPU, Dtype> buf, buf2;
};

#endif
