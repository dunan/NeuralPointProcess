#ifndef SYNTHETIC_H
#define SYNTHETIC_H

#include "data_loader.h"


void LoadSyntheticData()
{
    std::vector< std::vector<int> > raw_event_data;
    std::vector< std::vector<Dtype> > raw_time_data;

    LoadRawTimeEventData(raw_event_data, raw_time_data);
    
    // our simulation data only contains one line
    auto& time_label = raw_time_data[0];
    Dtype* time_data = new Dtype[time_label.size()]; 
    memcpy(time_data, time_label.data(), sizeof(Dtype) * time_label.size()); 

    auto& event_data = raw_event_data[0];    

    int data_len = time_label.size() - time_label.size() % (cfg::batch_size * cfg::bptt);    
    int seg_len = data_len / cfg::batch_size;
        
    int num_seg = seg_len / cfg::bptt;
    int test_len = (int)(num_seg * 0.1) * cfg::bptt;
    int train_len = seg_len - test_len;
        
    for (int i = time_label.size() - 1; i >= 1; --i)
        time_label[i] = time_label[i] - time_label[i - 1];
        
    Dtype* time_data_ptr = time_data; 

    if (cfg::T == 0)
    {
        for (int i = time_label.size() - 1; i >= 1; --i)
            time_data[i] = time_data[i] - time_data[i - 1];
    } else 
    {
        for (size_t i = 0; i < time_label.size(); ++i)
            time_data[i] = time_data[i] - (int)(time_data[i] / cfg::T) * cfg::T;
    }

    Dtype* time_label_ptr = time_label.data();

    int* event_data_ptr = event_data.data();
    for (unsigned i = 0; i < cfg::batch_size; ++i)
    {
        train_data->InsertSequence(event_data_ptr, 
                                   time_data_ptr, 
                                   time_label_ptr + 1, 
                                   train_len);
        event_data_ptr += train_len;
        time_data_ptr += train_len;
        time_label_ptr += train_len;
        test_data->InsertSequence(event_data_ptr, 
                                  time_data_ptr, 
                                  time_label_ptr + 1, 
                                  test_len);
        event_data_ptr += test_len;
        time_data_ptr += test_len;
        time_label_ptr += test_len;
    }

    delete[] time_data;
}


#endif