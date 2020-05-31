CUDA_PATH ?= /opt/cuda

HOST_ARCH := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
TARGET_SIZE := 64

HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 

CCFLAGS := -O2 -W -Wall -std=c++14
LDFLAGS :=

NV_CCFLAGS := -m${TARGET_SIZE} --expt-relaxed-constexpr
NV_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

NV_LDFLAGS := $(NV_CCFLAGS)
NV_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES := -I$(CUDA_PATH)/include
LIBRARIES :=  -lGL -lglut

SM ?= 61
GENCODE_FLAGS += -gencode arch=compute_$(SM),code=compute_$(SM)

all: build

build: schellingmodel_cuda

main.o:main.cpp
	$(HOST_COMPILER) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

model.o:model.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(NV_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

animation.o:animation.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(NV_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

schellingmodel_cuda: main.o model.o animation.o
	$(EXEC) $(NVCC) $(NV_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f schellingmodel_cuda schellingmodel_cuda.o

clobber: clean
