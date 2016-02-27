#include "config.h"
#include "user_net.h"

#include "data_loader.h"
#include "layer_holder.h"

GraphNN<mode, Dtype> gnn;
std::vector< GraphData<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;

std::vector<int> fathers;

ILayer<mode, Dtype>* AddRNNLayer(int time_step, 
							     Event& e, 
                                 ILayer<mode, Dtype>* event_feat, 
                                 ILayer<mode, Dtype>* time_feat)
{
	auto prev_self_hidden = layer_holder->GetCurState(e.uid);
	auto new_hidden = layer_holder->GetNewHidden(e.uid);
	new_hidden.first->AddParam(time_feat->name, param_dict["w_time2h"], GraphAtt::NODE); 
    new_hidden.first->AddParam(event_feat->name, param_dict["w_event2h"], GraphAtt::NODE);
	new_hidden.first->AddParam(prev_self_hidden.second->name, param_dict["w_h2h"], GraphAtt::NODE);
	gnn.AddEdge(time_feat, new_hidden.first);
    gnn.AddEdge(event_feat, new_hidden.first);
    gnn.AddEdge(prev_self_hidden.second, new_hidden.first);

	UserNet::GetFathers(e.uid, e.t, fathers);
	for (size_t i = 0; i < fathers.size(); ++i)
	{
		auto father_hidden = layer_holder->GetCurState(fathers[i]);
		new_hidden.first->AddParam(father_hidden.second->name, param_dict["w_father2h"], GraphAtt::NODE);	
		gnn.AddEdge(father_hidden.second, new_hidden.first);
	}    

	layer_holder->SetCurState(e.uid, new_hidden);
        
    gnn.AddEdge(new_hidden.first, new_hidden.second);
        
    return new_hidden.second;
}

void AddNetBlock(int time_step, Event& e)
{
	auto* time_input_layer =  layer_holder->GetStaticLayer(fmt::sprintf("time_input_%d", time_step));
	auto* event_input_layer = layer_holder->GetStaticLayer(fmt::sprintf("event_input_%d", time_step));
	auto* embed_layer = layer_holder->GetStaticLayer(fmt::sprintf("embed_%d", time_step));
	auto* relu_embed_layer = layer_holder->GetStaticLayer(fmt::sprintf("relu_embed_%d", time_step)); 
    gnn.AddEdge(event_input_layer, embed_layer);
    gnn.AddEdge(embed_layer, relu_embed_layer);

    ILayer<mode, Dtype>* recurrent_output = nullptr;
    if (cfg::gru)
    {
    } else
            recurrent_output = AddRNNLayer(time_step, e, relu_embed_layer, time_input_layer);

	auto* top_hidden = recurrent_output;
    if (cfg::n_h2)
    {
        auto* hidden_2 = layer_holder->GetStaticLayer(fmt::sprintf("hidden_2_%d", time_step));
        gnn.AddEdge(recurrent_output, hidden_2);
        auto* relu_2 =  layer_holder->GetStaticLayer(fmt::sprintf("relu_h2_%d", time_step));
        gnn.AddEdge(hidden_2, relu_2);
        top_hidden = relu_2;
    }
    auto* event_output_layer = layer_holder->GetStaticLayer(fmt::sprintf("event_out_%d", time_step));
    gnn.AddEdge(top_hidden, event_output_layer);

    auto* time_out_layer = layer_holder->GetStaticLayer(fmt::sprintf("time_out_%d", time_step));
    gnn.AddEdge(top_hidden, time_out_layer);

    auto* classnll = layer_holder->GetStaticLayer(fmt::sprintf("nll_%d", time_step));
    auto* mse_criterion = layer_holder->GetStaticLayer(fmt::sprintf("mse_%d", time_step));

    auto* mae_criterion = layer_holder->GetStaticLayer(fmt::sprintf("mae_%d", time_step));
    auto* err_cnt = layer_holder->GetStaticLayer(fmt::sprintf("err_cnt_%d", time_step));

 	gnn.AddEdge(time_out_layer, mse_criterion);
   	gnn.AddEdge(time_out_layer, mae_criterion);    	
    gnn.AddEdge(event_output_layer, classnll);
    gnn.AddEdge(event_output_layer, err_cnt);
}

void PrintResults(unsigned num_samples, std::map<std::string, Dtype>& loss_map, int unroll)
{
		Dtype rmse = 0.0, mae = 0.0, nll = 0.0, err_cnt = 0.0;
		for (int i = 0; i < unroll; ++i)
        {
            mae += loss_map[fmt::sprintf("mae_%d", i)];
            rmse += loss_map[fmt::sprintf("mse_%d", i)];
        	nll += loss_map[fmt::sprintf("nll_%d", i)]; 
            err_cnt += loss_map[fmt::sprintf("err_cnt_%d", i)];
        }
        rmse = sqrt(rmse / num_samples);
		mae /= num_samples;
		nll /= num_samples;
        err_cnt /= num_samples;
        std::cerr << fmt::sprintf("mae: %.4f\trmse: %.4f\tnll: %.4f\terr_rate: %.4f", mae, rmse, nll, err_cnt) << std::endl;
}

void Run(std::vector< Event >& dataset, unsigned unroll, Phase phase, int ttt)
{
	std::map<std::string, GraphData<mode, Dtype>* > feat, label;
	    for (unsigned i = 0; i < unroll; ++i)
    	{        
    		feat[fmt::sprintf("event_input_%d", i)] = g_event_input[i];
        	label[fmt::sprintf("nll_%d", i)] = g_event_label[i];
        	feat[fmt::sprintf("time_input_%d", i)] = g_time_input[i];
        	label[fmt::sprintf("mse_%d", i)] = g_time_label[i];
        	label[fmt::sprintf("mae_%d", i)] = g_time_label[i];
            label[fmt::sprintf("err_cnt_%d", i)] = g_event_label[i];        
    	}
	unsigned num_samples = 0;
	
	std::map<std::string, Dtype> test_loss_map;
	test_loss_map.clear();
	layer_holder->Reset();
	
	for (unsigned i = 0; i < dataset.size(); i += unroll)
	{
		if (i + unroll > dataset.size())
			break;
		num_samples += unroll;
		gnn.CleanLayers();
		LoadBatch(dataset, i, unroll, g_event_input, g_event_label, g_time_input, g_time_label);
		
		for (unsigned time_step = 0; time_step < unroll; ++time_step)
		{
			auto& e = dataset[i + time_step];
			AddNetBlock(time_step, e);
		}

		gnn.ForwardData(feat, phase);
		
        auto loss_map = gnn.ForwardLabel(label);
        
        for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
        {
            if (test_loss_map.count(it->first) == 0)
                    test_loss_map[it->first] = 0.0;
            test_loss_map[it->first] += it->second;
        }

        if (phase == TRAIN && ttt == 1)
        {
        	gnn.BackPropagation();
        	gnn.UpdateParams(cfg::lr, cfg::l2_penalty, cfg::momentum);  
        }

		layer_holder->CollectBack();	
		/*
if (ttt == -1)
			{
				std::cerr << i << std::endl;
			PrintResults(unroll, loss_map, unroll);
				for (auto it = gnn.layer_dict.begin(); it != gnn.layer_dict.end(); ++it)
				{
					auto* node_states = it->second->graph_output->node_states;
					if (node_states->GetMatType() == DENSE)
					{
						auto& state = node_states->DenseDerived();
						std::cerr << it->first << " " << state.Norm2() << std::endl;	
					}					
				}		
			}
			*/
		if (i / unroll % 10000 == 0)
		{
			//std::cerr << i << std::endl;
			//PrintResults(unroll, loss_map, unroll);			
		}
	}

	PrintResults(num_samples, test_loss_map, unroll);
}

int main(const int argc, const char** argv)
{
	cfg::LoadParams(argc, argv);
	GPUHandle::Init(0);
	LoadRawData();
	std::cerr << "totally " << cfg::num_users << " users" << std::endl;

	UserNet::Init(cfg::num_users, cfg::f_net);

	layer_holder = new LayerHolder<mode, Dtype>(cfg::num_users);
	SetupRNNParams();	
	InsertStaticRNNLayers();
	for (auto it = param_dict.begin(); it != param_dict.end(); ++it)
		gnn.AddParam(it->second);
	
	InitGraphData(g_event_input, g_event_label, g_time_input, g_time_label); 
	Run(train_data, cfg::bptt, TRAIN, 1);
	Run(test_data, 1, TEST, -1);
	Run(train_data, cfg::bptt, TRAIN, 1);
	Run(test_data, 1, TEST, -1);
	Run(train_data, cfg::bptt, TRAIN, 1);
	Run(test_data, 1, TEST, -1);
	Run(train_data, cfg::bptt, TRAIN, 1);
	Run(test_data, 1, TEST, -1);
	Run(train_data, cfg::bptt, TRAIN, 1);
	Run(test_data, 1, TEST, -1);
	/*
	Run(train_data, cfg::bptt, TRAIN);
	Run(test_data, 1, TEST);
	Run(train_data, cfg::bptt, TRAIN);
	Run(test_data, 1, TEST);
	Run(train_data, cfg::bptt, TRAIN);
	Run(test_data, 1, TEST);
	Run(train_data, cfg::bptt, TRAIN);
	Run(test_data, 1, TEST);
	//Run(train_data, cfg::bptt, TRAIN, 0);
*/
	GPUHandle::Destroy();
	return 0;	
}