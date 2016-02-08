#ifndef DATA_ADAPTER_H
#define DATA_ADAPTER_H

#include "data_loader.h"

inline void ProcessTimeDataLabel(std::vector<Dtype>& time_data, std::vector<Dtype>& time_label)
{
	time_data.clear();
	for (size_t i = 0; i < time_label.size(); ++i)
		time_data.push_back(time_label[i]);

	// we only predict the duration
	for (int i = time_label.size() - 1; i >= 1; --i)
        time_label[i] = time_label[i] - time_label[i - 1];

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

inline void LoadDataFromFile()
{
	std::vector< std::vector<int> > raw_event_data;
    std::vector< std::vector<Dtype> > raw_time_data;

    LoadRawTimeEventData(raw_event_data, raw_time_data);

    // only one sequence, manually make batches, synthetic data
    std::vector<Dtype> time_data;

    if (raw_time_data.size() == 1)
    {
    	    auto& time_label = raw_time_data[0];
			ProcessTimeDataLabel(time_data, time_label);

    		auto& event_data = raw_event_data[0];    

		    int data_len = time_label.size() - time_label.size() % (cfg::batch_size * cfg::bptt);    
    		int seg_len = data_len / cfg::batch_size;
        
    		int num_seg = seg_len / cfg::bptt;
    		int test_len = (int)(num_seg * cfg::test_pct) * cfg::bptt;
    		int train_len = seg_len - test_len;
                
    		Dtype* time_data_ptr = time_data.data(); 
    		Dtype* time_label_ptr = time_label.data();
    		int* event_data_ptr = event_data.data();

    		val_data->InsertSequence(event_data_ptr,
            		                 time_data_ptr,
                    		         time_label_ptr + 1,
                            		 101);

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
    } else { // multiple sequences, real data
    	std::cerr << raw_event_data.size() << " sequences loaded." << std::endl;
    	int train_seqs = 0, test_seqs = 0;
        for (unsigned i = 0; i < raw_event_data.size(); ++i)
        {
            assert(raw_event_data[i].size() == raw_time_data[i].size());
            for (size_t j = 0; j < raw_event_data[i].size(); ++j)
                assert(raw_event_data[i][j] >= 0);
        }
        if (!cfg::heldout_eval)
        {
            for (unsigned i = 0; i < raw_event_data.size(); ++i)
            {
                auto& time_label = raw_time_data[i];
                ProcessTimeDataLabel(time_data, time_label);

                int origin_len = raw_event_data[i].size();
                int test_len = origin_len * cfg::test_pct;
                int train_len = origin_len - test_len;

                if (test_len == 0 || train_len <= (int)cfg::bptt)
                {
                    std::cerr << "dropped short sequence in " << i << std::endl;
                    continue;
                }
                train_seqs++;
                test_seqs++;
                train_data->InsertSequence(raw_event_data[i].data(), 
                                           time_data.data(), 
                                           time_label.data() + 1, 
                                           train_len);
                test_len++;
                test_data->InsertSequence(raw_event_data[i].data() + train_len - 1, 
                                          time_data.data() + train_len - 1, 
                                          time_label.data() + train_len,
                                          test_len); 
            }
        } else 
        {
            int num_test_seqs = raw_event_data.size() * cfg::test_pct;
            int num_train_seqs = raw_event_data.size() - num_test_seqs;

            for (int i = 0; i < num_train_seqs; ++i)
            {
                auto& time_label = raw_time_data[i];
                ProcessTimeDataLabel(time_data, time_label);

                if (time_label.size() <= cfg::bptt)
                {
                    std::cerr << "dropped short sequence in " << i << std::endl;
                    continue;
                }
                train_seqs++;
                train_data->InsertSequence(raw_event_data[i].data(), 
                                           time_data.data(), 
                                           time_label.data() + 1, 
                                           raw_event_data[i].size());
            }

            for (unsigned i = num_train_seqs; i < raw_event_data.size(); ++i)
            {
                auto& time_label = raw_time_data[i];
                ProcessTimeDataLabel(time_data, time_label);

                test_seqs++;
                test_data->InsertSequence(raw_event_data[i].data(), 
                                          time_data.data(), 
                                          time_label.data() + 1, 
                                          raw_event_data[i].size());
            }
        }
    	
    	std::cerr << num_used << " sequences in use." << std::endl;
        std::cerr << "#train: " << train_data->num_samples << " #test: " << test_data->num_samples << std::endl;
    }
}

#endif