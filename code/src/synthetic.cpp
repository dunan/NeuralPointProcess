#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "dense_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "param_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "exp_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "batch_norm_param.h"
#include "config.h"
#include "data_loader.h"
#include "synthetic_data_loader.h"
#include "time_net.h"
#include "joint_net.h"
#include "event_net.h"

template<MatMode mode>
void Work()
{
    INet<mode, Dtype>* net; 

    switch (cfg::net_type)
    {
        case NetType::TIME:
            net = new TimeNet<mode, Dtype>();
            break;
        case NetType::JOINT:
            net = new JointNet<mode, Dtype>();
            break;
        case NetType::EVENT:
            net = new EventNet<mode, Dtype>();
            break;
        default:
            std::cerr << "unsupported nettype" << std::endl;
            return;
            break;
    }
    net->Setup();
    net->MainLoop();
}

int main(const int argc, const char** argv)
{	
    cfg::LoadParams(argc, argv);    
	GPUHandle::Init(cfg::dev_id);
    
    LoadSyntheticData();

    if (cfg::device_type == CPU)
        Work<CPU>();
    else
        Work<GPU>(); 

	GPUHandle::Destroy();
	return 0;
}
