#ifndef SYNTHETIC_H
#define SYNTHETIC_H

#include "data_loader.h"


void LoadSyntheticData()
{
    std::vector< std::vector<int> > raw_event_data;
    std::vector< std::vector<Dtype> > raw_time_data;

    LoadRawTimeEventData(raw_event_data, raw_time_data);
    
    // our simulation data only contains one line
    auto& time_data = raw_time_data[0];
    auto& event_data = raw_event_data[0];    

    int data_len = time_data.size() - time_data.size() % (cfg::batch_size * cfg::bptt);    
    int seg_len = data_len / cfg::batch_size;
        
    int num_seg = seg_len / cfg::bptt;
    int test_len = (int)(num_seg * 0.1) * cfg::bptt;
    int train_len = seg_len - test_len;
        
    for (int i = time_data.size() - 1; i >= 1; --i)
        time_data[i] = time_data[i] - time_data[i - 1];
        
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