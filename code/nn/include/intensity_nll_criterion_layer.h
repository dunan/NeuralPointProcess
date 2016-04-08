#ifndef INTENSITY_NLL_CRITERION_LAYER_H
#define INTENSITY_NLL_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "linear_param.h"

template<MatMode mode, typename Dtype>
class IntensityNllCriterionLayer : public ICriterionLayer<mode, Dtype>, public IParametric<mode, Dtype>
{
public:
			IntensityNllCriterionLayer(std::string _name, LinearParam<mode, Dtype>* _w, PropErr _properr = PropErr::T)
                : IntensityNllCriterionLayer<mode, Dtype>(_name, _w, 1.0, _properr) {}
                
			IntensityNllCriterionLayer(std::string _name, LinearParam<mode, Dtype>* _w, Dtype _lambda, PropErr _properr = PropErr::T);
             				 
            static std::string str_type();
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override; 
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;    
                         
            virtual bool HasParam() override { return true; } 
    
            virtual void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override;    
protected:
            LinearParam<mode, Dtype> *w;
            DenseMat<mode, Dtype> buffer;
            Dtype w_scalar;
};

#endif