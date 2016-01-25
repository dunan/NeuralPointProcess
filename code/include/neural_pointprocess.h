#ifndef NEURAL_POINTPROCESS_H
#define NEURAL_POINTPROCESS_H

std::vector< std::vector<int> > raw_event_data;
std::vector< std::vector<Dtype> > raw_time_data;

template<typename data_type>
inline void LoadRaw(const char* filename, std::vector< std::vector<data_type> >& raw_data)
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
    }       
}

inline void ReadRawData()
{
    std::cerr << "loading data..." << std::endl;
    assert(cfg::f_time_data && cfg::f_event_data);    
    
    LoadRaw(cfg::f_event_data, raw_event_data);
    LoadRaw(cfg::f_time_data, raw_time_data);
    assert(raw_event_data.size() == raw_time_data.size());
    std::set<int> label_set;
    label_set.clear();
    for (unsigned i = 0; i < raw_event_data.size(); ++i)
    {
        for (unsigned j = 0; j < raw_event_data[i].size(); ++j)
            label_set.insert(raw_event_data[i][j]);
    }
    std::cerr << "totally " << label_set.size() << " events" << std::endl;
    train_data = new DataLoader<TRAIN>(label_set.size(), cfg::batch_size); 
    test_data = new DataLoader<TEST>(label_set.size(), cfg::batch_size);
    
    for (unsigned i = 0; i < raw_event_data.size(); ++i)
    {
        assert(raw_event_data[i].size() == raw_time_data[i].size());
        
        int origin_len = raw_event_data[i].size();
        int test_len = origin_len * 0.1;
        int train_len = origin_len - test_len;
        train_len = 5;
        for (int j = 0; j < origin_len; ++j)
            raw_event_data[i][j]--; // the raw event is 1-based
        for (int j = origin_len - 1; j >= 1; --j)
            raw_time_data[i][j] = raw_time_data[i][j] - raw_time_data[i][j-1];     
        train_data->InsertSequence(raw_event_data[i].data(), raw_time_data[i].data(), train_len);
        test_len++;
        //test_data->InsertSequence(raw_event_data[i].data() + train_len - 1, raw_time_data[i].data() + train_len - 1, test_len); 
        test_data->InsertSequence(raw_event_data[i].data(), raw_time_data[i].data(), train_len);
        break;
    }
    std::cerr << raw_event_data.size() << " sequences loaded." << std::endl;
}

template<MatMode mode>
void InitGraphData(std::vector< GraphData<mode, Dtype>* >& g_event_input, 
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
        g_event_input.push_back(new GraphData<mode, Dtype>(SPARSE));
        g_event_label.push_back(new GraphData<mode, Dtype>(SPARSE));
        g_time_input.push_back(new GraphData<mode, Dtype>(DENSE));
        g_time_label.push_back(new GraphData<mode, Dtype>(DENSE));        
    }
}

#endif