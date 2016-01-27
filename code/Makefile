include $(HOME)/Workspace/cpp/graphnn/core_makefile

all: build/synthetic_time

CXXFLAGS += -I./include

build/synthetic_time: src/synthetic_time.cpp include/* $(gnn_objs) $(gnn_include)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS)

clean:
	rm build/*
