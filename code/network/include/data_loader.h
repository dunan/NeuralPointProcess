#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include "config.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <ctime>

struct Event
{
	int uid, m, target;
	long long t;
	Dtype dur;
	Event() {}
	Event(int _uid, int _m, int _target, long long _t, Dtype _dur) 
		: uid(_uid), m(_m), target(_target), t(_t), dur(_dur) {}

	bool operator<(const Event& other) const
	{
		return this->t < other.t;
	}
};

std::vector< Event > train_data, test_data;

template<typename data_type>
inline void ParseLine(std::string content, std::vector<data_type>& result)
{
	std::stringstream ss(content);
	data_type d;
	result.clear();
	while (ss >> d)
	{
		result.push_back(d);
	}
}

inline void LoadEvents(std::vector< Event >& dataset, std::string time_file, std::string event_file)
{
	dataset.clear();
	std::ifstream time_stream(time_file);
	std::ifstream event_stream(event_file);
	std::string read_buf;
	std::vector<long long> time_list;
	std::vector<int> marker_list;

	int cur_user = 0;
	while (getline(time_stream, read_buf))
	{
		ParseLine(read_buf, time_list);
		getline(event_stream, read_buf);
		ParseLine(read_buf, marker_list);
		assert(marker_list.size() == time_list.size());

		for (size_t i = 0; i + 1 < marker_list.size(); ++i)
		{
			Event e;
			e.uid = cur_user;
			e.m = marker_list[i]; 
			e.target = marker_list[i + 1];
			e.t = time_list[i];
			e.dur = cfg::time_scale * (time_list[i + 1] - time_list[i]);
			dataset.push_back(e);
		}
		cur_user++;
	}
	cfg::num_users = cur_user;
	sort(dataset.begin(), dataset.end());
}

inline void LoadRawData()
{
	train_data.clear();
	test_data.clear();
	assert(cfg::f_time_prefix && cfg::f_event_prefix);
	cfg::num_events = 2;

	LoadEvents(train_data, fmt::sprintf("%s-train.txt", cfg::f_time_prefix), fmt::sprintf("%s-train.txt", cfg::f_event_prefix)); 
	LoadEvents(test_data, fmt::sprintf("%s-test.txt", cfg::f_time_prefix), fmt::sprintf("%s-test.txt", cfg::f_event_prefix)); 

	std::cerr << "# train: " << train_data.size() << " # test: " << test_data.size() << std::endl;
}

inline void InitGraphData(std::vector< GraphData<mode, Dtype>* >& g_event_input, 
                   	   	  std::vector< GraphData<mode, Dtype>* >& g_event_label, 
                   		  std::vector< GraphData<mode, Dtype>* >& g_time_input, 
                   		  std::vector< GraphData<mode, Dtype>* >& g_time_label)
{
    g_event_input.clear();
    g_event_label.clear();
    g_time_input.clear();
    g_time_label.clear();
            
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {
        g_event_input.push_back(new GraphData<mode, Dtype>(DENSE));
        g_event_label.push_back(new GraphData<mode, Dtype>(SPARSE));     
        g_time_input.push_back(new GraphData<mode, Dtype>(SPARSE));        
        g_time_label.push_back(new GraphData<mode, Dtype>(DENSE));        
    }
}

inline void LoadBatch(std::vector< Event >& dataset, unsigned st, unsigned num, 
					  std::vector< GraphData<mode, Dtype>* >& g_event_input, 
                   	  std::vector< GraphData<mode, Dtype>* >& g_event_label, 
                   	  std::vector< GraphData<mode, Dtype>* >& g_time_input, 
                   	  std::vector< GraphData<mode, Dtype>* >& g_time_label)
{
	assert(mode == CPU);
	assert(st + num <= dataset.size());
	assert(num <= g_event_input.size());
	for (unsigned i = st; i < st + num; ++i)
	{
		auto& e = dataset[i];
		g_event_input[i - st]->graph->Resize(1, 1);
		g_event_label[i - st]->graph->Resize(1, 1);
		g_time_input[i - st]->graph->Resize(1, 1);
		g_time_label[i - st]->graph->Resize(1, 1);

		auto& event_feat = g_event_input[i - st]->node_states->DenseDerived();
		auto& event_label = g_event_label[i - st]->node_states->SparseDerived();
		auto& time_feat = g_time_input[i - st]->node_states->SparseDerived();
		auto& time_label = g_time_label[i - st]->node_states->DenseDerived();

		
		event_feat.Zeros(1, 2);
		event_feat.data[e.m] = 1.0;

		event_label.Resize(1, 2);
		event_label.ResizeSp(1, 2);

        event_label.data->ptr[0] = 0;
        event_label.data->ptr[1] = 1;
        event_label.data->col_idx[0] = e.target;
        event_label.data->val[0] = 1; 

		time_feat.Resize(1, cfg::time_dim);		
		time_feat.ResizeSp(cfg::unix_str.size(), 2);
		time_t tt = (time_t)e.t;
		struct tm *ptm = localtime(&tt);
		int cur_dim = 0, col;
		time_feat.data->ptr[0] = 0;
		time_feat.data->ptr[1] = cfg::unix_str.size();

		for (size_t j = 0; j < cfg::unix_str.size(); ++j)
        {
            switch (cfg::unix_str[j])
            {
                    case 'y':
                        col = ptm->tm_year;
                        break;
                    case 'm':
                        col = ptm->tm_mon;
                        break;
                    case 'd':
                        col = ptm->tm_mday - 1;
                        break;
                    case 'w':
                        col = ptm->tm_wday;
                        break;
                    case 'H':
                        col = ptm->tm_hour;
                        break;
                    case 'M':
                        col = ptm->tm_min;
                        break;
                    default:                        
                        assert(false);
                        break;
            }
            time_feat.data->col_idx[j] = col + cur_dim;
            time_feat.data->val[j] = 1.0;
            cur_dim += cfg::field_dim[cfg::unix_str[j]]; 
        }

        time_label.Resize(1, 1);
		time_label.data[0] = e.dur;
	}
}

#endif