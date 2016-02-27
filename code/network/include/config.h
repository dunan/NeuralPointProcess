#ifndef cfg_H
#define cfg_H

typedef double Dtype;

#include <iostream>
#include "cppformat/format.h"
#include <cstring>
#include <set>
#include <map>
#include "dense_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "param_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "mse_criterion_layer.h"
#include "abs_criterion_layer.h"
#include "classnll_criterion_layer.h"
#include "err_cnt_criterion_layer.h"
#include "sigmoid_layer.h"
#include "elewise_mul_layer.h"
#include "const_trans_layer.h"
#include "gather_layer.h"
#include "batch_norm_param.h"

const MatMode mode = CPU;

struct cfg
{
    static int iter;
    static unsigned bptt;
    static unsigned n_hidden; 
    static unsigned n_embed; 
    static unsigned num_events;
    static unsigned n_h2;
    static unsigned max_epoch; 
    static unsigned test_interval; 
    static unsigned report_interval; 
    static unsigned save_interval; 
    static unsigned time_dim;
    static unsigned num_users;
    static std::map<char, int> field_dim;
    static Dtype lambda;
    static Dtype lr;
    static Dtype l2_penalty; 
    static Dtype momentum; 
    static Dtype w_scale;
    static Dtype time_scale;
    static bool gru;
    static const char *f_time_prefix, *f_event_prefix, *f_net, *save_dir;
    static std::string unix_str;
    
    static void LoadParams(const int argc, const char** argv)
    {
        field_dim.clear();
        field_dim['y'] = 366;
        field_dim['m'] = 12;
        field_dim['d'] = 31;
        field_dim['w'] = 7;
        field_dim['H'] = 24;
        field_dim['M'] = 60;

        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-net") == 0)
                f_net = argv[i + 1];
            if (strcmp(argv[i], "-time") == 0)
                f_time_prefix = argv[i + 1];
            if (strcmp(argv[i], "-gru") == 0)
                gru = (bool)atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-unix_str") == 0)
                unix_str = std::string(argv[i + 1]); 
		    if (strcmp(argv[i], "-event") == 0)
		        f_event_prefix = argv[i + 1];
            if (strcmp(argv[i], "-t_scale") == 0)
                time_scale = atof(argv[i + 1]);
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);
            if (strcmp(argv[i], "-lambda") == 0)
                lambda = atof(argv[i + 1]);
            if (strcmp(argv[i], "-bptt") == 0)
                bptt = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-hidden") == 0)
			    n_hidden = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-embed") == 0)
			    n_embed = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-h2") == 0)
                n_h2 = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-maxe") == 0)
	       		max_epoch = atoi(argv[i + 1]);
		    if (strcmp(argv[i], "-int_test") == 0)
    			test_interval = atoi(argv[i + 1]);
    	   	if (strcmp(argv[i], "-int_report") == 0)
    			report_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-int_save") == 0)
    			save_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);
            if (strcmp(argv[i], "-w_scale") == 0)
                w_scale = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-m") == 0)
    			momentum = atof(argv[i + 1]);	
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
        }	

        assert(f_net && f_time_prefix && f_event_prefix); 
        assert(unix_str.size());
        time_dim = 0;
        for (size_t i = 0; i < unix_str.size(); ++i)
        {               
            assert(field_dim.count(unix_str[i])); 
            time_dim += field_dim[unix_str[i]];
        }
        std::cerr << "unix str: [" << unix_str << "] feature dimension of time: " << time_dim << std::endl;

        std::cerr << "gru = " << gru << std::endl;
        std::cerr << "n_h2 = " << n_h2 << std::endl;
        std::cerr << "time scale = " << time_scale << std::endl;
        std::cerr << "lambda = " << lambda << std::endl;
        std::cerr << "bptt = " << bptt << std::endl;
	    std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "n_embed = " << n_embed << std::endl;
        std::cerr << "max_epoch = " << max_epoch << std::endl;
    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
        std::cerr << "w_scale = " << w_scale << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
    }    
};

bool cfg::gru = false;
unsigned cfg::n_h2 = 0;
int cfg::iter = 0;
unsigned cfg::bptt = 3;
unsigned cfg::time_dim = 1;
unsigned cfg::num_users = 0;
unsigned cfg::num_events = 0;
unsigned cfg::n_hidden = 256;
unsigned cfg::n_embed = 128;
unsigned cfg::max_epoch = 200;
unsigned cfg::test_interval = 10000;
unsigned cfg::report_interval = 100;
unsigned cfg::save_interval = 50000; 
std::map<char, int> cfg::field_dim;
Dtype cfg::time_scale = 1.0;
Dtype cfg::lambda = 1.0;
Dtype cfg::lr = 0.0005;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
Dtype cfg::w_scale = 0.01;
const char* cfg::f_time_prefix = nullptr;
const char* cfg::f_event_prefix = nullptr;
const char* cfg::f_net = nullptr;
std::string cfg::unix_str = "";
const char* cfg::save_dir = "./saved";
#endif
