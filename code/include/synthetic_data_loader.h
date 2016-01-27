#ifndef SYNTHETIC_H
#define SYNTHETIC_H

template<typename data_type> 
void ReadDataSeq(const char* datafile, std::vector<data_type>& raw_data)
{
    raw_data.clear();
    std::ifstream raw_stream(datafile);
    data_type buf;
    while (raw_stream >> buf)
    {
        raw_data.push_back(buf);
    }
}

void ReadRaw()
{
    std::vector<Dtype> time_data;
    std::vector<int> event_data;

    ReadDataSeq(cfg::f_time_data, time_data);
    ReadDataSeq(cfg::f_event_data, event_data);
    
    for (unsigned i = 0; i < event_data.size(); ++i)
        event_data[i]--;

    int data_len = time_data.size() - time_data.size() % (cfg::batch_size * cfg::bptt);    
    int seg_len = data_len / cfg::batch_size;
        
    int num_seg = seg_len / cfg::bptt;
    int test_len = (int)(num_seg * 0.1) * cfg::bptt;
    int train_len = seg_len - test_len;
        
    for (int i = time_data.size() - 1; i >= 1; --i)
        time_data[i] = time_data[i] - time_data[i - 1];
        
    train_data = new DataLoader<TRAIN>(1, cfg::batch_size); 
    test_data = new DataLoader<TEST>(1, cfg::batch_size);

    Dtype* time_data_ptr = time_data.data();
    int* event_data_ptr = event_data.data();
    for (unsigned i = 0; i < cfg::batch_size; ++i)
    {
        train_data->InsertSequence(event_data_ptr, time_data_ptr, train_len);
        event_data_ptr += train_len;
        time_data_ptr += train_len;
        test_data->InsertSequence(event_data_ptr, time_data_ptr, test_len);
        event_data_ptr += test_len;
        time_data_ptr += test_len;
    }
}


#endif