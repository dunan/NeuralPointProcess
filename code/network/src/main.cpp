#include "config.h"
#include "user_net.h"

#include "data_loader.h"
#include "layer_holder.h"

GraphNN<mode, Dtype> gnn;
std::vector< GraphData<mode, Dtype>* > g_event_input, g_event_label, g_time_input, g_time_label;
std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label, test_feat, test_label;

void LinkTrainData() 
{
    	for (unsigned i = 0; i < cfg::bptt; ++i)
    	{        
    		train_feat[fmt::sprintf("event_input_%d", i)] = g_event_input[i];
        	train_label[fmt::sprintf("nll_%d", i)] = g_event_label[i];
        	train_feat[fmt::sprintf("time_input_%d", i)] = g_time_input[i];
        	train_label[fmt::sprintf("mse_%d", i)] = g_time_label[i];
        	train_label[fmt::sprintf("mae_%d", i)] = g_time_label[i];
            train_label[fmt::sprintf("err_cnt_%d", i)] = g_event_label[i];        
    	}
}

void LinkTestData()
{
		test_feat["event_input_0"] = g_event_input[0];
		test_feat["time_input_0"] = g_time_input[0];
		test_label["mse_0"] = g_time_label[0];
		test_label["mae_0"] = g_time_label[0];
		test_label["nll_0"] = g_event_label[0];
        test_label["err_cnt_0"] = g_event_label[0];       
}

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

void Run(std::vector< Event >& dataset, unsigned unroll, Phase phase)
{
	unsigned num_samples = 0;
	layer_holder->Reset();
	std::map<std::string, Dtype> test_loss_map;
	test_loss_map.clear();
	for (unsigned i = 0; i < dataset.size(); i += unroll)
	{
		std::cerr << i << std::endl;
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

		if (phase == TRAIN)
			gnn.ForwardData(train_feat, TRAIN);
		else
			gnn.ForwardData(test_feat, TEST);
        auto loss_map = gnn.ForwardLabel(test_label);
        for (auto it = loss_map.begin(); it != loss_map.end(); ++it)
        {
            if (test_loss_map.count(it->first) == 0)
                    test_loss_map[it->first] = 0.0;
            test_loss_map[it->first] += it->second;
        }

		layer_holder->CollectBack();	
		break;
	}	
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
	LinkTrainData();
	LinkTestData();
	Run(train_data, cfg::bptt, TRAIN);
	//Run(train_data, 1);

	GPUHandle::Destroy();
	return 0;	
}