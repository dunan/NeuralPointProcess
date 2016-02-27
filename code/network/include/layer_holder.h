#ifndef LAYER_HOLDER_H
#define LAYER_HOLDER_H

#include "config.h"
#include <deque>

template<MatMode mode, typename Dtype>
class LayerHolder
{
public:

	LayerHolder(unsigned _num_users) : num_users(_num_users)
	{
		layer_dict.clear();
		cur_hidden_states.clear();
		hidden_pool.clear();
		for (size_t i = 0; i < num_users; ++i)
		{
			cur_hidden_states.push_back(std::deque< std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > >());
			hidden_pool.push_back(std::deque< std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > >());
		}
	}

	ILayer<mode, Dtype>* GetStaticLayer(std::string layer_name)
	{
		assert(layer_dict.count(layer_name));
		return layer_dict[layer_name];
	}

	void InsertLayer(ILayer<mode, Dtype>* layer)
	{
		assert(layer_dict.count(layer->name) == 0);
		layer_dict[layer->name] = layer;
	}

	std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > GetCurState(int user_id)
	{
		assert (user_id >= 0 && user_id < num_users);
		if (cur_hidden_states[user_id].size() == 0)
			cur_hidden_states[user_id].push_back(CreateNewState(user_id)); 

		return cur_hidden_states[user_id].front();
	}

	void SetCurState(int user_id, std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > state)
	{
		assert (user_id >= 0 && user_id < num_users);
		auto& cur_deque = cur_hidden_states[user_id];
		for (size_t i = 0; i < cur_deque.size(); ++i)
			assert(state.first->name != cur_deque[i].first->name);

		cur_deque.push_front(state);
	}

	std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > GetNewHidden(int user_id)
	{
		assert (user_id >= 0 && user_id < num_users);

		if (hidden_pool[user_id].size() == 0)
			hidden_pool[user_id].push_back(CreateNewState(user_id));

		auto state = hidden_pool[user_id].front();
		hidden_pool[user_id].pop_front();
		return state;
	}

	void CleanSeq(std::deque< std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > >& qq)
	{
		if (qq.size())
		{
			for (unsigned i = 0; i < qq.size(); ++i)
			{
				auto& pair_state = qq[i];
				auto& linear = pair_state.first->graph_output->node_states->DenseDerived();
				linear.Zeros(1, cfg::n_hidden);
				auto& act = pair_state.second->graph_output->node_states->DenseDerived();
				act.Zeros(1, cfg::n_hidden);
			}
		}
	}

	void Reset()
	{		
		for (unsigned i = 0; i < num_users; ++i)
		{
			CleanSeq(cur_hidden_states[i]);
			CleanSeq(hidden_pool[i]);
		}
		CollectBack();
	}

	void CollectBack()
	{
		for (unsigned i = 0; i < num_users; ++i)
		{
			auto& cur_deque = cur_hidden_states[i];
			if (cur_deque.size())
			{
				for (size_t j = 1; j < cur_deque.size(); ++j)
					hidden_pool[i].push_back(cur_deque[j]);

				auto cur_state = cur_deque[0];
				cur_deque.clear();
				cur_deque.push_back(cur_state);
			}
		}
	}

	std::map< std::string, ILayer<mode, Dtype>* > layer_dict;

	std::vector< std::deque< std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > > > cur_hidden_states, hidden_pool;

	unsigned num_users;

protected:

	std::pair< NodeLayer<mode, Dtype>*, ReLULayer<mode, Dtype>* > CreateNewState(int user_id)
	{
		int cur_num = cur_hidden_states[user_id].size() + hidden_pool[user_id].size();
		auto* layer = new NodeLayer<mode, Dtype>(fmt::sprintf("user_%d_hidden_%d", user_id, cur_num));
		layer->graph_output->graph->Resize(1, 1);
		auto& states = layer->graph_output->node_states->DenseDerived();
		states.Zeros(1, cfg::n_hidden);

		auto* act = new ReLULayer<mode, Dtype>(fmt::sprintf("user_%d_hidden_%d_relu", user_id, cur_num), 
											   GraphAtt::NODE, WriteType::INPLACE); 
		act->graph_output->graph->Resize(1, 1);
		auto& act_states = act->graph_output->node_states->DenseDerived();
		act_states.Zeros(1, cfg::n_hidden);

		return std::make_pair(layer, act);
	}
};

LayerHolder<mode, Dtype>* layer_holder;
std::map< std::string, IParam<mode, Dtype>* > param_dict;

inline void SetupRNNParams()
{
		param_dict["w_embed"] = new LinearParam<mode, Dtype>("w_embed", cfg::num_events, cfg::n_embed, 0, cfg::w_scale);
    	param_dict["w_event2h"] = new LinearParam<mode, Dtype>("w_event2h", cfg::n_embed, cfg::n_hidden, 0, cfg::w_scale);
		param_dict["w_time2h"] = new LinearParam<mode, Dtype>("w_time2h", cfg::time_dim, cfg::n_hidden, 0, cfg::w_scale);
    	param_dict["w_h2h"] = new LinearParam<mode, Dtype>("w_h2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);

    	param_dict["w_father2h"] = new LinearParam<mode, Dtype>("w_father2h", cfg::n_hidden, cfg::n_hidden, 0, cfg::w_scale);
        unsigned hidden_size = cfg::n_hidden;
        if (cfg::n_h2)
        {
            hidden_size = cfg::n_h2;
            param_dict["w_hidden2"] = new LinearParam<mode, Dtype>("w_hidden2", cfg::n_hidden, cfg::n_h2, 0, cfg::w_scale);
        }
        param_dict["w_event_out"] = new LinearParam<mode, Dtype>("w_event_out", hidden_size, cfg::num_events, 0, cfg::w_scale);
    	param_dict["w_time_out"] = new LinearParam<mode, Dtype>("w_time_out", hidden_size, 1, 0, cfg::w_scale);	
}

inline void InsertStaticRNNLayers()
{
	for (int time_step = 0; time_step < (int)cfg::bptt; ++time_step)
	{
		layer_holder->InsertLayer(new InputLayer<mode, Dtype>(fmt::sprintf("time_input_%d", time_step), GraphAtt::NODE));
		layer_holder->InsertLayer(new InputLayer<mode, Dtype>(fmt::sprintf("event_input_%d", time_step), GraphAtt::NODE));
		layer_holder->InsertLayer(new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("embed_%d", time_step), param_dict["w_embed"], GraphAtt::NODE));
		layer_holder->InsertLayer(new ReLULayer<mode, Dtype>(fmt::sprintf("relu_embed_%d", time_step), GraphAtt::NODE, WriteType::INPLACE));
        if (cfg::n_h2)
        {
            layer_holder->InsertLayer(new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("hidden_2_%d", time_step), param_dict["w_hidden2"], GraphAtt::NODE));
            layer_holder->InsertLayer(new ReLULayer<mode, Dtype>(fmt::sprintf("relu_h2_%d", time_step), GraphAtt::NODE, WriteType::INPLACE));
        }
        layer_holder->InsertLayer(new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("event_out_%d", time_step), param_dict["w_event_out"], GraphAtt::NODE));
        layer_holder->InsertLayer(new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("time_out_%d", time_step), param_dict["w_time_out"], GraphAtt::NODE));
        layer_holder->InsertLayer(new ClassNLLCriterionLayer<mode, Dtype>(fmt::sprintf("nll_%d", time_step), true));
        layer_holder->InsertLayer(new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step), cfg::lambda));
        layer_holder->InsertLayer(new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N));
        layer_holder->InsertLayer(new ErrCntCriterionLayer<mode, Dtype>(fmt::sprintf("err_cnt_%d", time_step)));
	}
}

#endif