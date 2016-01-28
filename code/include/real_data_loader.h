#ifndef NEURAL_POINTPROCESS_H
#define NEURAL_POINTPROCESS_H

#include "data_loader.h"

inline void LoadRealData()
{
    std::vector< std::vector<int> > raw_event_data;
    std::vector< std::vector<Dtype> > raw_time_data;

    LoadRawTimeEventData(raw_event_data, raw_time_data);
    
    for (unsigned i = 0; i < raw_event_data.size(); ++i)
    {
        assert(raw_event_data[i].size() == raw_time_data[i].size());
        
        int origin_len = raw_event_data[i].size();
        int test_len = origin_len * 0.1;
        int train_len = origin_len - test_len;
        for (int j = 0; j < origin_len; ++j)
            raw_event_data[i][j]--; // the raw event is 1-based
        for (int j = origin_len - 1; j >= 1; --j)
            raw_time_data[i][j] = raw_time_data[i][j] - raw_time_data[i][j-1];     
        train_data->InsertSequence(raw_event_data[i].data(), raw_time_data[i].data(), train_len);
        test_len++;
        test_data->InsertSequence(raw_event_data[i].data() + train_len - 1, raw_time_data[i].data() + train_len - 1, test_len); 
    }
    std::cerr << raw_event_data.size() << " sequences loaded." << std::endl;
}

#endif