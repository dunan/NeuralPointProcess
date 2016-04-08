#include "dur_pred_layer.h"
#include "linear_param.h"

template<MatMode mode, typename Dtype>
DurPredLayer<mode, Dtype>::DurPredLayer(std::string _name, LinearParam<mode, Dtype>* _w)
							: ILayer<mode, Dtype>(_name, PropErr::N), w(_w)
{
		this->state = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
void DurPredLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
		assert(operands.size() == 1);       

}

template class DurPredLayer<CPU, float>;
template class DurPredLayer<CPU, double>;
template class DurPredLayer<GPU, float>;
template class DurPredLayer<GPU, double>;