include $(HOME)/Workspace/cpp/graphnn/core_makefile

all: build/time_series

build/time_series: time_series.cpp $(gnn_objs) $(gnn_include)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS) 

clean:
	rm build/*
