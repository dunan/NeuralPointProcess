include $(HOME)/Workspace/cpp/graphnn/core_makefile

all: build/neural_pointprocess build/synthetic build/neural_pp_time

CXXFLAGS += -I./include

build/neural_pp_time: src/neural_pp_time.cpp include/* $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

build/neural_pointprocess: src/neural_pointprocess.cpp include/* $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

build/synthetic: src/synthetic.cpp include/config.h include/data_loader.h $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

clean:
	rm build/*
