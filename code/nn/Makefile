GNN_HOME=../graphnn-1.11

include $(GNN_HOME)/make_common

lib_dir := $(GNN_HOME)/build/lib
gnn_lib := $(lib_dir)/libgnn.a

include_dirs := $(CUDA_HOME)/include $(MKL_ROOT)/include $(GNN_HOME)/include/matrix $(GNN_HOME)/include/graphnn ./include
CXXFLAGS += $(addprefix -I,$(include_dirs))

all: build/main

cpp_files := $(shell $(FIND) src/lib -name "*.cpp" -printf "%P\n")
cxx_obj_files := $(subst .cpp,.o,$(cpp_files))

objs := $(addprefix build/lib/,$(cxx_obj_files))
DEPS := $(objs:.o=.d)

build/main: src/main.cpp $(gnn_lib) $(objs)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) -L$(lib_dir) -lgnn $(LDFLAGS)

DEPS += build/main.d

build/lib/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
