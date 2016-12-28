# NeuralPointProcess

#### Prerequisites

Tested under Ubuntu 14.04

##### Download and install cuda from https://developer.nvidia.com/cuda-toolkit

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    
  in .bashrc, add the following path (suppose you installed to the default path)
  
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
##### Download and install intel mkl

  in .bashrc, add the following path
  
    source {path_to_your_intel_root/name_of_parallel_tool_box}/bin/psxevars.sh
    export MKL_ROOT={path_to_your_intel_root}/mkl
    
##### Install cppformat (now called fmtlib)

    check https://github.com/fmtlib/fmt for help
    
##### Build static graphnn v1.11 library

    navigate to code/graphnn-1.11
    modify configurations in make_common file
    make
    
#### Build main nn code

    navigate to code/nn
    make
    
#### run test

    navigate to code/nn
    ./synthetic_run.sh

#### reproduce the results reported in paper

    navigate to code/nn
    modify the scripts under code/nn/scripts (or code/nn/server_scripts, the two folders 
    have different parameter settings and path configurations)
    execute the script
    
#### about data
    all the synthetic datasets have been pushed to this repo. Some large datasets are not available here. 
    
    
    
    
