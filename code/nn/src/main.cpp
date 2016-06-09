#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "config.h"
#include "data_loader.h"
#include "data_adapter.h"
#include "time_net.h"
#include "joint_net.h"
#include "event_net.h"

const MatMode mode = CPU;

template<MatMode mode>
void Work()
{
    INet<mode, Dtype>* net; 
    IEventTimeLoader<mode>* etloader = nullptr;
    if (cfg::unix_time)
        etloader = new UnixTimeLoader<mode>();
    else 
        etloader = new SingleTimeLoader<mode>();
    switch (cfg::net_type)
    {
        case NetType::TIME:
            net = new TimeNet<mode, Dtype>(etloader);
            break;      
        case NetType::JOINT:
            net = new JointNet<mode, Dtype>(etloader);
            break;
        case NetType::EVENT:
            net = new EventNet<mode, Dtype>(etloader);
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
    
    LoadDataFromFile();

    if (cfg::device_type == CPU)
        Work<CPU>();
    else
        Work<GPU>(); 

	GPUHandle::Destroy();
	return 0;
}
