#include "dur_pred_layer.h"
#include "linear_param.h"

template<MatMode mode, typename Dtype>
DurPredLayer<mode, Dtype>::DurPredLayer(std::string _name, LinearParam<mode, Dtype>* _w)
							: ILayer<mode, Dtype>(_name, PropErr::N), w(_w)
{
		this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
void DurPredLayer<mode, Dtype>::PT(Dtype t)
{
		auto& buf = this->grad->DenseDerived();
		buf.Resize(e_vt_hj.rows, 1);

		buf.Fill(t * w_scalar);
		buf.Exp();
		buf.EleWiseMul(e_vt_hj);
		
		buffer.CopyFrom(e_vt_hj);
		buffer.Axpy(-1.0, buf);
		buffer.Scale(1.0 / w_scalar);
		buffer.Exp();
		buffer.EleWiseMul(buf);
		buffer.Scale(t);		
}

template<MatMode mode, typename Dtype>
void DurPredLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
		assert(operands.size() == 1);       
		w_scalar = w->p["weight"]->value.AsScalar();
		e_vt_hj.Exp(operands[0]->state->DenseDerived());

		auto& int_result = this->state->DenseDerived();

		int precision = 1000;
		int n = 2 * precision;
		Dtype T = 100;
		Dtype delta = T / n;
		// f(a) + f(b)
		PT(T);
		int_result.CopyFrom(buffer);

		for (int i = 1; i < precision; ++i)
		{
			Dtype x = (2 * i - 1) * delta;

			// 4 * f(x)
			PT(x);
			buffer.Scale(4.0);
			int_result.Axpy(1.0, buffer);
			// 2 * f(x + delta)
			PT(x + delta);
			buffer.Scale(2.0);
			int_result.Axpy(1.0, buffer);		
		}

		// 4 ∗ f (b − delta) 
		PT(T - delta);
		buffer.Scale(4.0);
		int_result.Axpy(1.0, buffer);

		int_result.Scale(delta / 3.0);
}


template class DurPredLayer<CPU, float>;
template class DurPredLayer<CPU, double>;
template class DurPredLayer<GPU, float>;
template class DurPredLayer<GPU, double>;