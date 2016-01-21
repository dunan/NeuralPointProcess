include $(HOME)/Workspace/cpp/graphnn/core_makefile

all: build/neural_pointprocess build/synthetic

CXXFLAGS += -I./include

build/neural_pointprocess: src/neural_pointprocess.cpp include/* $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

build/synthetic: src/synthetic.cpp include/* $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

clean:
	rm build/*
