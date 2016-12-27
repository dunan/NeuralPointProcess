#ifndef cfg_H
#define cfg_H

typedef double Dtype;
#include "fmt/format.h"
#include <set>
#include "imatrix.h"
#include <map>

enum class NetType
{
    TIME = 0,
    EVENT = 1,
    JOINT = 2
};

enum class LossType
{
    MSE = 0,
    EXP = 1,
    INTENSITY = 2
};

struct cfg
{
    static int dev_id, iter, test_top;
    static unsigned bptt;     
    static unsigned n_hidden; 
    static unsigned n_embed; 
    static unsigned n_h2;
    static unsigned batch_size; 
    static unsigned max_epoch; 
    static unsigned test_interval; 
    static unsigned report_interval; 
    static unsigned save_interval; 
    static unsigned time_dim;
    static std::map<char, int> field_dim;
    static NetType net_type;
    static LossType loss_type;
    static Dtype lambda;
    static Dtype lr;
    static Dtype l2_penalty; 
    static Dtype momentum; 
    static MatMode device_type;
    static Dtype w_scale;
    static Dtype T;
    static Dtype time_scale;
    static bool save_eval, has_eval, unix_time, gru, use_history;
    static const char *f_time_prefix, *f_event_prefix, *save_dir;
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
            if (strcmp(argv[i], "-time") == 0)
                f_time_prefix = argv[i + 1];
            if (strcmp(argv[i], "-mode") == 0)
		    {
		        if (strcmp(argv[i + 1], "CPU") == 0)
		            device_type = CPU;
		        else if (strcmp(argv[i + 1], "GPU") == 0)
		            device_type = GPU;
		        else throw "unknown device"; 
                std::cerr << "device_type = " << argv[i + 1] << std::endl;
		    }
            if (strcmp(argv[i], "-net") == 0)
            {
                if (strcmp(argv[i + 1], "time") == 0)
                    net_type = NetType::TIME;
                else if (strcmp(argv[i + 1], "event") == 0)
                    net_type = NetType::EVENT;
                else if (strcmp(argv[i + 1], "joint") == 0)
                    net_type = NetType::JOINT;
                else throw "unknown net type"; 
                std::cerr << "net_type = " << argv[i + 1] << std::endl;
            }
            if (strcmp(argv[i], "-loss") == 0)
            {
                if (strcmp(argv[i + 1], "mse") == 0)
                    loss_type = LossType::MSE;
                else if (strcmp(argv[i + 1], "exp") == 0)
                    loss_type = LossType::EXP;
                else if (strcmp(argv[i + 1], "intensity") == 0)
                    loss_type = LossType::INTENSITY;
                else throw "unknown net type"; 
                std::cerr << "loss_type = " << argv[i + 1] << std::endl;
            }
            if (strcmp(argv[i], "-save_eval") == 0)
                save_eval = (bool)atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-eval") == 0)
                has_eval = (bool)atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-unix") == 0)
                unix_time = (bool)atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-history") == 0)
                use_history = (bool)atoi(argv[i + 1]); 
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
            if (strcmp(argv[i], "-T") == 0)
                T = atof(argv[i + 1]);
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
            if (strcmp(argv[i], "-test_top") == 0)
                test_top = atoi(argv[i + 1]);             
		    if (strcmp(argv[i], "-b") == 0)
    			batch_size = atoi(argv[i + 1]);
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
            if (strcmp(argv[i], "-device") == 0)
    			dev_id = atoi(argv[i + 1]);
        }	

        if (unix_time)
        {
            assert(unix_str.size());
            time_dim = 0;
            for (size_t i = 0; i < unix_str.size(); ++i)
            {               
                assert(field_dim.count(unix_str[i])); 
                time_dim += field_dim[unix_str[i]];
            }
            std::cerr << "unix str: [" << unix_str << "] feature dimension of time: " << time_dim << std::endl;
        }

        std::cerr << "use_history = " << use_history << std::endl;
        std::cerr << "gru = " << gru << std::endl;
        std::cerr << "n_h2 = " << n_h2 << std::endl;
        std::cerr << "time scale = " << time_scale << std::endl;
        std::cerr << "lambda = " << lambda << std::endl;
        std::cerr << "unix_time = " << unix_time << std::endl;
        std::cerr << "bptt = " << bptt << std::endl;
	    std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "n_embed = " << n_embed << std::endl;
        std::cerr << "T = " << T << std::endl; 
        std::cerr << "batch_size = " << batch_size << std::endl;
        std::cerr << "max_epoch = " << max_epoch << std::endl;
    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;
        std::cerr << "save_eval = " << save_eval << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
        std::cerr << "w_scale = " << w_scale << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
        std::cerr << "device id = " << dev_id << std::endl;    
    }    
};

bool cfg::gru = false;
unsigned cfg::n_h2 = 0;
int cfg::dev_id = 0;
int cfg::iter = 0;
unsigned cfg::bptt = 3;
unsigned cfg::time_dim = 1;
unsigned cfg::n_hidden = 256;
unsigned cfg::n_embed = 128;
unsigned cfg::batch_size = 50;
unsigned cfg::max_epoch = 200;
unsigned cfg::test_interval = 10000;
unsigned cfg::report_interval = 100;
unsigned cfg::save_interval = 50000; 
std::map<char, int> cfg::field_dim;
int cfg::test_top = -1;
Dtype cfg::time_scale = 1.0;
Dtype cfg::lambda = 1.0;
Dtype cfg::T = 0;
Dtype cfg::lr = 0.0005;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
Dtype cfg::w_scale = 0.01;
MatMode cfg::device_type = GPU;
bool cfg::use_history = false;
bool cfg::save_eval = false;
bool cfg::has_eval = false;
bool cfg::unix_time = false;
const char* cfg::f_time_prefix = nullptr;
const char* cfg::f_event_prefix = nullptr;
std::string cfg::unix_str = "";
const char* cfg::save_dir = "./saved";
NetType cfg::net_type = NetType::TIME;
LossType cfg::loss_type = LossType::MSE;

#endif
