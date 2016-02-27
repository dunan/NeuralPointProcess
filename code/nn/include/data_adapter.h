#ifndef DATA_ADAPTER_H
#define DATA_ADAPTER_H

#include "data_loader.h"
#include <map>

template<MatMode mode>
inline void InitGraphData(std::vector< IMatrix<mode, Dtype>* >& g_event_input, 
                   std::vector< IMatrix<mode, Dtype>* >& g_event_label, 
                   std::vector< IMatrix<mode, Dtype>* >& g_time_input, 
                   std::vector< IMatrix<mode, Dtype>* >& g_time_label)
{
    g_event_input.clear();
    g_event_label.clear();
    g_time_input.clear();
    g_time_label.clear();
            
    for (unsigned i = 0; i < cfg::bptt; ++i)
    {
        g_event_input.push_back(new SparseMat<mode, Dtype>());
        g_event_label.push_back(new SparseMat<mode, Dtype>());
        if (cfg::unix_time)
            g_time_input.push_back(new SparseMat<mode, Dtype>());
        else
            g_time_input.push_back(new DenseMat<mode, Dtype>());
        g_time_label.push_back(new DenseMat<mode, Dtype>());        
    }
}

inline void ProcessTimeDataLabel(std::vector<Dtype>& time_data, std::vector<Dtype>& time_label)
{
	time_data.clear();
	for (size_t i = 0; i < time_label.size(); ++i)
		time_data.push_back(time_label[i]);

	// we only predict the duration
	for (int i = time_label.size() - 1; i >= 1; --i)
        time_label[i] = cfg::time_scale * (time_label[i] - time_label[i - 1]);

    if (cfg::unix_time)
        return;
    if (cfg::T == 0)
    {
        for (int i = time_label.size() - 1; i >= 1; --i)
            time_data[i] = time_data[i] - time_data[i - 1];
    } else 
    {
        for (size_t i = 0; i < time_label.size(); ++i)
            time_data[i] = time_data[i] - (int)(time_data[i] / cfg::T) * cfg::T;
    }
}

template<typename data_type>
inline void LoadRaw(const char* filename, std::vector< std::vector<data_type> >& raw_data, int max_seq_num)
{
    raw_data.clear();
    std::ifstream f_stream(filename);
    std::string read_buf;
    data_type d;
    while (getline(f_stream, read_buf))
    {
        std::stringstream ss(read_buf);
        std::vector<data_type> cur_seq;
        cur_seq.clear();
        while (ss >> d)
        {
            cur_seq.push_back(d); 
        }
        raw_data.push_back(cur_seq);
        if (max_seq_num >= 0 && raw_data.size() >= max_seq_num)
            break;
    }
}

inline size_t GetNumEvents(std::vector< std::vector<int> >& raw_event_train, 
                           std::vector< std::vector<int> >& raw_event_test)
{
    std::map<int, int> label_map;
    int num_events = 0;
    
    for (unsigned i = 0; i < raw_event_train.size() + raw_event_test.size(); ++i)
    {
        auto& event_data = i < raw_event_train.size() ? raw_event_train[i] : raw_event_test[i - raw_event_train.size()];
        for (unsigned j = 0; j < event_data.size(); ++j)
        {
            if (label_map.count(event_data[j]) == 0)
            {
                label_map[event_data[j]] = num_events;
                num_events++;
            }
            event_data[j] = label_map[event_data[j]];
        }
    }
    assert(num_events == (int)label_map.size());
    std::cerr << "totally " << label_map.size() << " events" << std::endl;
    return label_map.size();
}

template<Phase phase>
inline void Insert2Loader(DataLoader<phase>* dataset, 
                          std::vector< std::vector<int> >& raw_event_data, 
                          std::vector< std::vector<Dtype> >& raw_time_data, 
                          size_t min_len)
{
    std::vector<Dtype> time_data;
    for (size_t i = 0; i < raw_event_data.size(); ++i)
    {
        assert(raw_event_data[i].size() == raw_time_data[i].size());
        auto& time_label = raw_time_data[i];
        ProcessTimeDataLabel(time_data, time_label);

        if (time_label.size() <= min_len)
        {
            std::cerr << "dropped short sequence in " << i << std::endl;
            continue;
        }
        dataset->InsertSequence(raw_event_data[i].data(), 
                                time_data.data(), 
                                time_label.data() + 1, 
                                raw_event_data[i].size());

        if (phase == TEST && i == 0 && cfg::has_eval)
        {
            val_data->InsertSequence(raw_event_data[i].data(),
                                     time_data.data(),
                                     time_label.data() + 1,
                                     raw_event_data[i].size());
        }
    }
}

inline void LoadDataFromFile()
{
	std::vector< std::vector<int> > raw_event_train, raw_event_test;
    std::vector< std::vector<Dtype> > raw_time_train, raw_time_test;

    std::cerr << "loading data..." << std::endl;
    assert(cfg::f_time_prefix && cfg::f_event_prefix);   

    LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_event_prefix).c_str(), raw_event_train, -1);
    LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_time_prefix).c_str(), raw_time_train, -1);
    //LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_event_prefix).c_str(), raw_event_train, cfg::test_top);
    //LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_time_prefix).c_str(), raw_time_train, cfg::test_top);

    //LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_event_prefix).c_str(), raw_event_test, cfg::test_top);    
    //LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_time_prefix).c_str(), raw_time_test, cfg::test_top);
    LoadRaw(fmt::sprintf("%s-test.txt", cfg::f_event_prefix).c_str(), raw_event_test, cfg::test_top);
    LoadRaw(fmt::sprintf("%s-test.txt", cfg::f_time_prefix).c_str(), raw_time_test, cfg::test_top);

    assert(raw_event_train.size() == raw_time_train.size());
    assert(raw_event_test.size() == raw_time_test.size());

    size_t num_events = GetNumEvents(raw_event_train, raw_event_test);
    train_data = new DataLoader<TRAIN>(num_events, cfg::batch_size); 
    test_data = new DataLoader<TEST>(num_events, cfg::batch_size);
    val_data = new DataLoader<TEST>(num_events, 1);

    Insert2Loader(train_data, raw_event_train, raw_time_train, cfg::bptt);
    Insert2Loader(test_data, raw_event_test, raw_time_test, 1);

    std::cerr << "#train: " << train_data->num_samples << " #test: " << test_data->num_samples << std::endl;
    if (cfg::has_eval)
        std::cerr << "#eval: " << val_data->num_samples << std::endl;
}

#endif