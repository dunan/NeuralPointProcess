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

inline size_t GetNumEvents(std::vector< std::vector<int> >& raw_event_train, 
                           std::vector< std::vector<int> >& raw_event_test)
{
    std::set<int> label_set;
    int min_id = 1000000000, max_id = 0;
    for (unsigned i = 0; i < raw_event_train.size() + raw_event_test.size(); ++i)
    {
        auto& event_data = i < raw_event_train.size() ? raw_event_train[i] : raw_event_test[i - raw_event_train.size()];
        for (unsigned j = 0; j < event_data.size(); ++j)
        {
            event_data[j]--;
            label_set.insert(event_data[j]);
            if (event_data[j] < min_id)
                min_id = event_data[j];
            if (event_data[j] > max_id)
                max_id = event_data[j];
        }
    }
    assert(min_id == 0 && max_id + 1 == (int)label_set.size());
    std::cerr << "totally " << label_set.size() << " events" << std::endl;
    return label_set.size();
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
    }
}

inline void LoadDataFromFile()
{
	std::vector< std::vector<int> > raw_event_train, raw_event_test;
    std::vector< std::vector<Dtype> > raw_time_train, raw_time_test;

    std::cerr << "loading data..." << std::endl;
    assert(cfg::f_time_prefix && cfg::f_event_prefix);   

    LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_event_prefix).c_str(), raw_event_train);
    LoadRaw(fmt::sprintf("%s-test.txt", cfg::f_event_prefix).c_str(), raw_event_test);
    LoadRaw(fmt::sprintf("%s-train.txt", cfg::f_time_prefix).c_str(), raw_time_train);
    LoadRaw(fmt::sprintf("%s-test.txt", cfg::f_time_prefix).c_str(), raw_time_test);

    assert(raw_event_train.size() == raw_time_train.size());
    assert(raw_event_test.size() == raw_time_test.size());

    size_t num_events = GetNumEvents(raw_event_train, raw_event_test);
    train_data = new DataLoader<TRAIN>(num_events, cfg::batch_size); 
    test_data = new DataLoader<TEST>(num_events, cfg::batch_size);
    val_data = new DataLoader<TEST>(num_events, 1);

    Insert2Loader(train_data, raw_event_train, raw_time_train, cfg::bptt);
    Insert2Loader(test_data, raw_event_test, raw_time_test, 1);
    std::cerr << "#train: " << train_data->num_samples << " #test: " << test_data->num_samples << std::endl;
/*
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
        
    */
}

#endif