#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "graph_data.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <set>
#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "mkl_helper.h"
#include "cuda_helper.h"

class IDataLoader
{
public:
   
    IDataLoader(size_t _num_events, size_t _batch_size) : num_events(_num_events), batch_size(_batch_size)
    {
        event_sequences.clear();
        time_sequences.clear();
        cursors.resize(batch_size);
        index_pool.clear();
        num_samples = 0;
        initialized = false;
    }
    
    inline void InsertSequence(int* event_seq, Dtype* time_seq, int seq_len)
    {
        num_samples += seq_len - 1;
        InsertSequence(event_seq, event_sequences, seq_len);
        InsertSequence(time_seq, time_sequences, seq_len);   
    }
    
    virtual void StartNewEpoch()
    {
        initialized = true;
        if (index_pool.size() != event_sequences.size())
        {
            index_pool.clear();
            assert(event_sequences.size() == time_sequences.size()); 
            for (unsigned i = 0; i < event_sequences.size(); ++i)
            {
                index_pool.push_back(i);                     
            }
        }
        for (unsigned i = 0; i < batch_size; ++i)
        {
            cursors[i].first = index_pool.front();
            cursors[i].second = 0;
            index_pool.pop_front();
        }
    }
    
    size_t num_samples, num_events, batch_size; 

private:
    void ReloadSlot(unsigned batch_idx)    
    {
        index_pool.push_back(cursors[batch_idx].first); 
        cursors[batch_idx].first = index_pool.front();
        cursors[batch_idx].second = 0;
        index_pool.pop_front();
    }

protected:


    template<typename data_type>
    inline void InsertSequence(data_type* seq, std::vector< std::vector<data_type> >& sequences, int seq_len)
    {
        std::vector<data_type> cur_seq;
        cur_seq.clear();
        for (int i = 0; i < seq_len; ++i)
            cur_seq.push_back(seq[i]);
        sequences.push_back(cur_seq);
    }   

    void ReloadSlot(GraphData<CPU, Dtype>* g_last_hidden, unsigned batch_idx)
    {
        auto& last_hidden = g_last_hidden->node_states->DenseDerived();
        memset(last_hidden.data + last_hidden.cols * batch_idx, 0, sizeof(Dtype) * last_hidden.cols);

        ReloadSlot(batch_idx);
    }    

    void ReloadSlot(GraphData<GPU, Dtype>* g_last_hidden, unsigned batch_idx)
    {
        auto& last_hidden = g_last_hidden->node_states->DenseDerived();
        cudaMemset(last_hidden.data + last_hidden.cols * batch_idx, 0, sizeof(Dtype) * last_hidden.cols); 

        ReloadSlot(batch_idx);
    }
    
    template<MatMode mode>
    void LoadEvent(GraphData<mode, Dtype>* g_feat, GraphData<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step)
    {
        g_feat->graph->Resize(1, cur_batch_size);
        g_label->graph->Resize(1, cur_batch_size);

        auto& feat = g_feat->node_states->SparseDerived();
        auto& label = g_label->node_states->SparseDerived();
        
        event_feat_cpu.Resize(cur_batch_size, num_events);
        event_feat_cpu.ResizeSp(cur_batch_size, cur_batch_size + 1); 
        
        event_label_cpu.Resize(cur_batch_size, num_events);
        event_label_cpu.ResizeSp(cur_batch_size, cur_batch_size + 1);
        
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            event_feat_cpu.data->ptr[i] = i;
            event_feat_cpu.data->col_idx[i] = event_sequences[cursors[i].first][cursors[i].second + step]; 
            event_feat_cpu.data->val[i] = 1;
            
            event_label_cpu.data->ptr[i] = i;
            event_label_cpu.data->col_idx[i] = event_sequences[cursors[i].first][cursors[i].second + step + 1];
            event_label_cpu.data->val[i] = 1;                        
        }
        event_feat_cpu.data->ptr[cur_batch_size] = cur_batch_size;
        event_label_cpu.data->ptr[cur_batch_size] = cur_batch_size;
        
        feat.CopyFrom(event_feat_cpu);
        label.CopyFrom(event_label_cpu);
    } 
    
    template<MatMode mode>
    void LoadTime(GraphData<mode, Dtype>* g_feat, GraphData<mode, Dtype>* g_label, unsigned cur_batch_size, unsigned step)
    {
        g_feat->graph->Resize(1, cur_batch_size);
        g_label->graph->Resize(1, cur_batch_size);
        auto& feat = g_feat->node_states->DenseDerived();
        auto& label = g_label->node_states->DenseDerived();
        
        time_feat_cpu.Resize(cur_batch_size, 1);
        time_label_cpu.Resize(cur_batch_size, 1);
        
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            time_feat_cpu.data[i] = time_sequences[cursors[i].first][cursors[i].second + step];
            time_label_cpu.data[i] = time_sequences[cursors[i].first][cursors[i].second + step + 1];
        }
        
        feat.CopyFrom(time_feat_cpu);
        label.CopyFrom(time_label_cpu);
    } 
   
    bool initialized;
    std::vector< std::pair<unsigned, unsigned> > cursors;                 
    std::vector< std::vector<int> > event_sequences;
    std::vector< std::vector<Dtype> > time_sequences;
    std::deque< unsigned > index_pool;
    SparseMat<CPU, Dtype> event_feat_cpu, event_label_cpu;
    DenseMat<CPU, Dtype> time_feat_cpu, time_label_cpu;
};


template<Phase phase>
class DataLoader; 

template<>
class DataLoader<TRAIN> : public IDataLoader
{
public:

    DataLoader(unsigned _num_events, unsigned _batch_size) : IDataLoader(_num_events, _batch_size)
    {
        
    }
    
    template<MatMode mode>             
    inline void NextBpttBatch(int bptt, GraphData<mode, Dtype>* g_last_hidden,
                              std::vector< GraphData<mode, Dtype>* >& g_event_input,
                              std::vector< GraphData<mode, Dtype>* >& g_time_input, 
                              std::vector< GraphData<mode, Dtype>* >& g_event_label,
                              std::vector< GraphData<mode, Dtype>* >& g_time_label)
    {
        if (!initialized)
            this->StartNewEpoch();

        for (unsigned i = 0; i < this->batch_size; ++i)
        {
            // need to load a new sequences                                   
            if (cursors[i].second + bptt >= event_sequences[cursors[i].first].size())
            {                              
                this->ReloadSlot(g_last_hidden, i); 
            }
        }
        g_last_hidden->graph->Resize(1, this->batch_size);
        for (int j = 0; j < bptt; ++j)
        {                                  
            this->LoadEvent(g_event_input[j], g_event_label[j], this->batch_size, j);                        
            this->LoadTime(g_time_input[j], g_time_label[j], this->batch_size, j);           
        }        
        for (unsigned i = 0; i < this->batch_size; ++i)
            cursors[i].second += bptt;           
    }
}; 

template<>
class DataLoader<TEST> : public IDataLoader
{
public:
    
    DataLoader(unsigned _num_events, unsigned _batch_size) : IDataLoader(_num_events, _batch_size)
    {
        available.clear();
    }    
    
    template<MatMode mode>
    inline bool NextBatch(GraphData<mode, Dtype>* g_last_hidden,
                          GraphData<mode, Dtype>* g_event_input, 
                          GraphData<mode, Dtype>* g_time_input, 
                          GraphData<mode, Dtype>* g_event_label, 
                          GraphData<mode, Dtype>* g_time_label)
    {
        if (!this->initialized)
            this->StartNewEpoch();
        
        unsigned delta_size = 0;                    
        for (unsigned i = 0; i < cur_batch_size; ++i)
        {
            // need to load a new sequences                                   
            if (cursors[i].second + 1 >= event_sequences[cursors[i].first].size())
            {                
                if (index_pool.size() > 0 && available[index_pool.front()])
                {   
                    available[index_pool.front()] = false;                       
                    this->ReloadSlot(g_last_hidden, i);
                } else 
                    delta_size++;
            }
        }

        if (cur_batch_size == delta_size)
            return false;
        
        if (delta_size)
        {
            auto& prev_hidden = g_last_hidden->node_states->DenseDerived();    
            if (cur_batch_size == batch_size) // insufficient for the first time
            {
                std::vector<unsigned> ordered; 
                for (unsigned i = 0; i < batch_size; ++i)
                    ordered.push_back(i);

                for (unsigned i = 0; i < batch_size - 1; ++i)
                    for (unsigned j = i + 1; j < batch_size; ++j)
                    {
                        if (cursors[j].second + 1 >= event_sequences[cursors[j].first].size())
                            continue;  // no need to move forward 

                        // if x is full, or y is longer than x
                        if (cursors[i].second + 1 >= event_sequences[cursors[i].first].size() || 
                            event_sequences[cursors[j].first].size() - cursors[j].second > 
                            event_sequences[cursors[i].first].size() - cursors[i].second )
                        {
                            unsigned tmp = ordered[i];
                            ordered[i] = ordered[j];
                            ordered[j] = tmp;
                            auto t = cursors[i];
                            cursors[i] = cursors[j];
                            cursors[j] = t;
                        }
                    }
                DenseMat<mode, Dtype> buf(batch_size - delta_size, prev_hidden.cols);

                for (unsigned i = 0; i < buf.rows; ++i)
                {
                    cudaMemcpy(buf.data + i * buf.cols, prev_hidden.data + ordered[i] * buf.cols, sizeof(Dtype) * buf.cols, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToDevice); 
                }  
                prev_hidden.CopyFrom(buf);     
            } else
                prev_hidden.Resize(cur_batch_size - delta_size, prev_hidden.cols);
            cur_batch_size -= delta_size;
        }
        g_last_hidden->graph->Resize(1, cur_batch_size);
        this->LoadEvent(g_event_input, g_event_label, cur_batch_size, 0);
        this->LoadTime(g_time_input, g_time_label, cur_batch_size, 0);
        for (unsigned i = 0; i < cur_batch_size; ++i)
            cursors[i].second++;         
        return true;
    }

    virtual void StartNewEpoch() override
    {        
        IDataLoader::StartNewEpoch();
        if (available.size() != event_sequences.size())
            available.resize(event_sequences.size());
        
        for (unsigned i = 0; i < available.size(); ++i)
            available[i] = true;
               
        for (unsigned i = 0; i < this->batch_size; ++i)
            available[cursors[i].first] = false;

        cur_batch_size = batch_size;
    }
    
protected:    
    unsigned cur_batch_size;
    std::vector<bool> available;             
}; 

DataLoader<TRAIN>* train_data;
DataLoader<TEST>* test_data;


#endif