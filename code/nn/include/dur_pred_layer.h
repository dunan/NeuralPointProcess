#ifndef DUR_PRED_LAYER_H
#define DUR_PRED_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class LinearParam; 

template<MatMode mode, typename Dtype>
class DurPredLayer : public ILayer<mode, Dtype>
{
public:
			DurPredLayer(std::string _name, LinearParam<mode, Dtype>* _w);
                            
			virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override; 
            
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override {}

protected:
            LinearParam<mode, Dtype> *w;
            Dtype w_scalar;            
};

#endif