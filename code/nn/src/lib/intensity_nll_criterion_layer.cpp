#include "intensity_nll_criterion_layer.h"

template<MatMode mode, typename Dtype>
IntensityNllCriterionLayer<mode, Dtype>::IntensityNllCriterionLayer(std::string _name, LinearParam<mode, Dtype>* _w, Dtype _lambda, PropErr _properr)
                                    : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
{
        this->w = _w;
                
        this->grad = new DenseMat<mode, Dtype>();    
        this->state = new DenseMat<mode, Dtype>();    
}

template<MatMode mode, typename Dtype>
std::string IntensityNllCriterionLayer<mode, Dtype>::str_type()
{
        return "LambdaTNLL";
}

template<MatMode mode, typename Dtype>
void IntensityNllCriterionLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
        // operands[0]: v^T h_j; operands[1]: dt = t_{j+1} - t_j
        assert(operands.size() == 2);       
        
        w_scalar = w->p["weight"]->value.AsScalar();       
        auto& vt_hj = operands[0]->state->DenseDerived();                
        auto& cur_state = this->state->DenseDerived();
        
        // cur_state = v^T h_j + b + w^T dt;
        cur_state.CopyFrom(vt_hj);                        
        w->UpdateOutput(operands[1]->state, &cur_state, 1.0, phase);
                        
        // exp(v^th_j + b)
        buffer.Exp(vt_hj);
        
        auto& grad = this->grad->DenseDerived();        
        grad.Exp(cur_state);
        grad.Axpy(-1.0, buffer);
        grad.Scale(1.0 / w_scalar);
        
        buffer.GeaM(1.0, Trans::N, grad, -1.0, Trans::N, cur_state);        
                        
        this->loss = buffer.Sum();                                                                                                                  
}

template<MatMode mode, typename Dtype>
void IntensityNllCriterionLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
        assert(operands.size() == 2 && cur_idx == 0);
        
        auto& prev_grad = operands[0]->grad->DenseDerived();
        auto& grad = this->grad->DenseDerived();
        
        Dtype batch_size = grad.rows;
        if (beta == 0)
        {
            prev_grad.CopyFrom(grad);
            prev_grad.Scale(this->lambda / batch_size);
        }            
        else
            prev_grad.Axpby(this->lambda / batch_size, grad, beta);
        prev_grad.Add(-this->lambda / batch_size);
}

template<MatMode mode, typename Dtype>
void IntensityNllCriterionLayer<mode, Dtype>::AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx)
{
        assert(operands.size() == 2);
        if (cur_idx)
            return;

        auto& delta_t = operands[1]->state->DenseDerived();
        auto& grad = this->grad->DenseDerived();
        auto& cur_state = this->state->DenseDerived();
        
        Dtype dw = -delta_t.Sum();
        dw -= grad.Sum() / w_scalar;
        cur_state.Exp();        
        dw += cur_state.Dot(delta_t) / w_scalar;
        dw *= this->lambda / grad.rows;

        this->w->p["weight"]->grad.Add(dw); 
}
    
template class IntensityNllCriterionLayer<CPU, float>;
template class IntensityNllCriterionLayer<CPU, double>;
template class IntensityNllCriterionLayer<GPU, float>;
template class IntensityNllCriterionLayer<GPU, double>;