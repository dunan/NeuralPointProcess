#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "dense_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "node_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "exp_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "simple_node_layer.h"
#include "batch_norm_param.h"

typedef double Dtype;
const MatMode mode = GPU;
int dev_id = 0, iter = 0;
unsigned bptt = 3, n_recur_layers = 1;

unsigned n_hidden = 128;
unsigned batch_size = 50;
unsigned max_epoch = 200;
unsigned test_interval = 10000;
unsigned report_interval = 100;
unsigned save_interval = 50000;
Dtype lr = 0.0005;
Dtype l2_penalty = 0;
Dtype momentum = 0;

const char *datafile, *save_dir = "./saved";

GraphNN<mode, Dtype> net_train, net_test;

GraphData<mode, Dtype>* g_last_hidden_train, *g_last_hidden_test;
std::vector< GraphData<mode, Dtype>* > g_inputs, g_labels;

class DataLoader
{
public:

    DataLoader(int _b_size, int _seg_len) : b_size(_b_size), seg_len(_seg_len)
    {
        data_buf = new Dtype[(long long)b_size * (long long)_seg_len];
        label_buf = new Dtype[(long long)b_size * (long long)_seg_len];
        num_seq = cur_pos = 0;
        x_cpu.Resize(b_size, 1);
        y_cpu.Resize(b_size, 1);
    }
    
    void AddSeq(const Dtype* data, const Dtype* label, int len)
    {
        int offset = num_seq;
        for (int i = 0; i < len; ++i)
        {
            data_buf[offset] = data[i];
            label_buf[offset] = label[i];
            offset += b_size;
        }
        num_seq++;
    }        
    
    void NextBatch(GraphData<mode, Dtype>* x, GraphData<mode, Dtype>* y)
    {
        x->graph->Resize(1, b_size);
        y->graph->Resize(1, b_size);
        memcpy(x_cpu.data, data_buf + cur_pos * b_size, sizeof(Dtype) * b_size);
        memcpy(y_cpu.data, label_buf + cur_pos * b_size, sizeof(Dtype) * b_size);
        auto& node_states = x->node_states->DenseDerived();
        node_states.CopyFrom(x_cpu);
        auto& label_states = y->node_states->DenseDerived();
        label_states.CopyFrom(y_cpu);
        
        cur_pos++;
        if (cur_pos == seg_len)
            cur_pos = 0;
    }
        
    const int b_size, seg_len;
    DenseMat<CPU, Dtype> x_cpu, y_cpu;
    int num_seq, cur_pos;
    Dtype* data_buf, *label_buf;    
};

DataLoader* train_data, *test_data, *val_train_data;

void LoadParams(const int argc, const char** argv)
{
	datafile = argv[1];
	for (int i = 2; i < argc; i += 2)
	{
		if (strcmp(argv[i], "-lr") == 0)
			lr = atof(argv[i + 1]);
        if (strcmp(argv[i], "-bptt") == 0)
			bptt = atoi(argv[i + 1]);                                    
		if (strcmp(argv[i], "-cur_iter") == 0)
			iter = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-h") == 0)
			n_hidden = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-b") == 0)
			batch_size = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-maxe") == 0)
			max_epoch = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-int_test") == 0)
			test_interval = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-int_report") == 0)
			report_interval = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-int_save") == 0)
			save_interval = atoi(argv[i + 1]);
		if (strcmp(argv[i], "-l2") == 0)
			l2_penalty = atof(argv[i + 1]);
		if (strcmp(argv[i], "-m") == 0)
			momentum = atof(argv[i + 1]);	
		if (strcmp(argv[i], "-svdir") == 0)
			save_dir = argv[i + 1];
        if (strcmp(argv[i], "-device") == 0)
			dev_id = atoi(argv[i + 1]);
	}
	
    std::cerr << "bptt = " << bptt << std::endl;
	std::cerr << "n_hidden = " << n_hidden << std::endl;
	std::cerr << "batch_size = " << batch_size << std::endl;
	std::cerr << "max_epoch = " << max_epoch << std::endl;
	std::cerr << "test_interval = " << test_interval << std::endl;
	std::cerr << "report_interval = " << report_interval << std::endl;
	std::cerr << "save_interval = " << save_interval << std::endl;
	std::cerr << "lr = " << lr << std::endl;
	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
	std::cerr << "momentum = " << momentum << std::endl;
	std::cerr << "init iter = " << iter << std::endl;	
    std::cerr << "device id = " << dev_id << std::endl;    
}

ILayer<mode, Dtype>* AddNetBlocks(int time_step, GraphNN<mode, Dtype>& gnn, ILayer<mode, Dtype> *last_hidden_layer, LinearParam<mode, Dtype>* i2h, LinearParam<mode, Dtype>* h2h, LinearParam<mode, Dtype>* h2o)
{    
    gnn.AddLayer(last_hidden_layer);
    
    auto* input_layer = new InputLayer<mode, Dtype>(fmt::sprintf("input_%d", time_step));
    auto* hidden_layer = new NodeLayer<mode, Dtype>(fmt::sprintf("hidden_%d", time_step));
    hidden_layer->AddParam(input_layer->name, i2h);
    hidden_layer->AddParam(last_hidden_layer->name, h2h);                
    auto* relu_layer = new ReLULayer<mode, Dtype>(fmt::sprintf("reluact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);
    auto* output_layer = new SimpleNodeLayer<mode, Dtype>(fmt::sprintf("out_%d", time_step), h2o);
    auto* exp_layer = new ExpLayer<mode, Dtype>(fmt::sprintf("expact_%d", time_step), WriteType::INPLACE, ActTarget::NODE);                                
        
    auto* mse_criterion = new MSECriterionLayer<mode, Dtype>(fmt::sprintf("mse_%d", time_step));
    auto* mae_criterion = new ABSCriterionLayer<mode, Dtype>(fmt::sprintf("mae_%d", time_step), PropErr::N);    	
            	
    gnn.AddLayer(input_layer);
    gnn.AddLayer(hidden_layer);
    gnn.AddLayer(relu_layer);        
    gnn.AddLayer(output_layer);
    gnn.AddLayer(exp_layer);
    gnn.AddLayer(mse_criterion);
        
    gnn.AddEdge(input_layer, hidden_layer);
    gnn.AddEdge(last_hidden_layer, hidden_layer);
    gnn.AddEdge(hidden_layer, relu_layer);
    gnn.AddEdge(relu_layer, output_layer);
    gnn.AddEdge(output_layer, exp_layer);        
    gnn.AddEdge(exp_layer, mse_criterion);    
    gnn.AddEdge(exp_layer, mae_criterion);
        
    return relu_layer;   
}

void InitNetTrain()
{    
    g_last_hidden_train = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_train->node_states->DenseDerived().Zeros(batch_size, n_hidden);
    g_last_hidden_test = new GraphData<mode, Dtype>(DENSE);
    g_last_hidden_test->node_states->DenseDerived().Zeros(batch_size, n_hidden);
    
    g_inputs.clear();
    g_labels.clear();        
    for (unsigned i = 0; i < bptt; ++i)
    {
        g_inputs.push_back(new GraphData<mode, Dtype>(DENSE));
        g_labels.push_back(new GraphData<mode, Dtype>(DENSE));
    }
        
    auto* i2h = new LinearParam<mode, Dtype>("i2h", 1, n_hidden, 0, 0.01);
    auto* h2h = new LinearParam<mode, Dtype>("h2h", n_hidden, n_hidden, 0, 0.01);
    auto* h2o = new LinearParam<mode, Dtype>("h2o", n_hidden, 1, 0, 0.01);
    
    net_train.AddParam(i2h);
    net_train.AddParam(h2h);
    net_train.AddParam(h2o);
    
    ILayer<mode, Dtype>* last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden_train");    
    for (unsigned i = 0; i < bptt; ++i)
    {
        auto* new_hidden = AddNetBlocks(i, net_train, last_hidden_layer, i2h, h2h, h2o);
        last_hidden_layer = new_hidden;
    }
        
    net_test.AddParam(i2h);
    net_test.AddParam(h2h);
    net_test.AddParam(h2o);        
    auto* test_last_hidden_layer = new InputLayer<mode, Dtype>("last_hidden_test");
    AddNetBlocks(0, net_test, test_last_hidden_layer, i2h, h2h, h2o);            
}

void ReadRaw()
{
    std::vector<Dtype> raw_data;
    raw_data.clear();
    std::ifstream raw_stream(datafile);
    Dtype buf;
    while (raw_stream >> buf)
    {
        raw_data.push_back(buf);
    }
    
    int data_len = raw_data.size() - raw_data.size() % (batch_size * bptt);    
    int seg_len = data_len / batch_size;
        
    int num_seg = seg_len / bptt;    
    int test_len = (int)(num_seg * 0.1) * bptt;
    int train_len = seg_len - test_len;
        
    for (int i = raw_data.size() - 1; i >= 1; --i)
        raw_data[i] = raw_data[i] - raw_data[i - 1];
        
    Dtype* data_ptr = raw_data.data(), *label_ptr = raw_data.data() + 1;

    train_data = new DataLoader(batch_size, train_len);
    test_data = new DataLoader(batch_size, test_len);
    
    val_train_data = new DataLoader(1, 100);
    val_train_data->AddSeq(data_ptr, label_ptr, 100); 
    
    for (unsigned i = 0; i < batch_size; ++i)
    {
        train_data->AddSeq(data_ptr, label_ptr, train_len);
        data_ptr += train_len;
        label_ptr += train_len;
        test_data->AddSeq(data_ptr, label_ptr, test_len);
        data_ptr += test_len;
        label_ptr += test_len;
    }
}

DenseMat<CPU, Dtype> output_buf;

int main(const int argc, const char** argv)
{	
    LoadParams(argc, argv);    
	GPUHandle::Init(dev_id);
    
    ReadRaw();    
    InitNetTrain();
    
    int max_iter = (long long)max_epoch * train_data->seg_len / bptt;
    int init_iter = iter;
    if (init_iter > 0)
	{
		printf("loading model for iter=%d\n", init_iter);
		net_train.Load(fmt::sprintf("%s/iter_%d.model", save_dir, init_iter));
	}
    
    Dtype mae, rmse;
    auto& last_hidden_train = g_last_hidden_train->node_states->DenseDerived();
    
    std::map<std::string, GraphData<mode, Dtype>* > train_feat, train_label;
    train_feat["last_hidden_train"] = g_last_hidden_train;
    for (unsigned i = 0; i < bptt; ++i)
    {        
        train_feat[fmt::sprintf("input_%d", i)] = g_inputs[i];
        train_label[fmt::sprintf("mse_%d", i)] = g_labels[i];
        train_label[fmt::sprintf("mae_%d", i)] = g_labels[i];
    }
    
    for (; iter <= max_iter; ++iter)
	{
		if (iter % test_interval == 0)
		{
			std::cerr << "testing" << std::endl;
            rmse = mae = 0.0;
            auto& last_hidden_test = g_last_hidden_test->node_states->DenseDerived(); 
            last_hidden_test.Zeros(batch_size, n_hidden);               
			for (int i = 0; i < test_data->seg_len; ++i)
			{                
                test_data->NextBatch(g_inputs[0], g_labels[0]);                
             	net_test.ForwardData({{"input_0", g_inputs[0]}, {"last_hidden_test", g_last_hidden_test}}, TEST);                                
				auto loss_map = net_test.ForwardLabel({{"mse_0", g_labels[0]}, {"mae_0", g_labels[0]}});
				rmse += loss_map["mse_0"];
				mae += loss_map["mae_0"];
                
                net_test.GetDenseNodeState("reluact_0", last_hidden_test);
			}
            rmse = sqrt(rmse / test_data->seg_len / batch_size);
			mae /= test_data->seg_len * batch_size;
			std::cerr << fmt::sprintf("test mae: %.4f\t test rmse: %.4f", mae, rmse) << std::endl;
            
            
            last_hidden_test.Zeros(1, n_hidden);
            rmse = mae = 0.0;
            FILE* test_pred_fid = fopen("100_test.txt", "w");               
			for (int i = 0; i < val_train_data->seg_len; ++i)
			{                
                val_train_data->NextBatch(g_inputs[0], g_labels[0]);                
             	net_test.ForwardData({{"input_0", g_inputs[0]}, {"last_hidden_test", g_last_hidden_test}}, TEST);                                
				auto loss_map = net_test.ForwardLabel({{"mse_0", g_labels[0]}, {"mae_0", g_labels[0]}});
				rmse += loss_map["mse_0"];
				mae += loss_map["mae_0"];                
                net_test.GetDenseNodeState("reluact_0", last_hidden_test);
                net_test.GetDenseNodeState("expact_0", output_buf);
                for (unsigned j = 0; j < output_buf.rows; ++j)
                    fprintf(test_pred_fid, "%.6f\n", output_buf.data[j]);
			}             
            
            fclose(test_pred_fid);
                                            
            rmse = sqrt(rmse / val_train_data->seg_len);
			mae /= val_train_data->seg_len; 
			std::cerr << fmt::sprintf("val_train_100 mae: %.4f\t val_train_100 rmse: %.4f", mae, rmse) << std::endl;            
		}
		
		if (iter % save_interval == 0 && iter != init_iter)
		{			
			printf("saving model for iter=%d\n", iter);			
			net_train.Save(fmt::sprintf("%s/iter_%d.model", save_dir, iter));
		}		
        
        if (train_data->cur_pos == 0)
        {
            last_hidden_train.Zeros(batch_size, n_hidden);            
        }

        for (unsigned i = 0; i < bptt; ++i)
        { 
            train_data->NextBatch(g_inputs[i], g_labels[i]);                            
        }
        
        net_train.ForwardData(train_feat, TRAIN);        
        auto loss_map = net_train.ForwardLabel(train_label);
        //net_train.GetDenseNodeState(fmt::sprintf("reluact_%d", bptt - 1), last_hidden_train);
        
        if (iter % report_interval == 0)
		{
            mae = rmse = 0.0;
            for (unsigned i = 0; i < bptt; ++i)
            {
                mae += loss_map[fmt::sprintf("mae_%d", i)];
                rmse += loss_map[fmt::sprintf("mse_%d", i)];  
            }
            rmse = sqrt(rmse / bptt / batch_size);
			mae /= bptt * batch_size;
			std::cerr << fmt::sprintf("train iter=%d\tmae: %.4f\trmse: %.4f", iter, mae, rmse) << std::endl;        	
		}
        
        net_train.BackPropagation();
		net_train.UpdateParams(lr, l2_penalty, momentum);	
	}
    
	GPUHandle::Destroy();
	return 0;
}
