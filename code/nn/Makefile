include $(HOME)/Workspace/cpp/graphnn/core_makefile

all: build/main

CXXFLAGS += -I./include

build/main: src/main.cpp include/* $(gnn_objs) $(gnn_include)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.o %.cuo, $^) $(LDFLAGS)

clean:
	rm build/*
