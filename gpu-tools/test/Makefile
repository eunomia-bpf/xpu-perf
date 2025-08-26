# Auto-detect CUDA installation
CUDA_PATH_12_8 := /usr/local/cuda-12.8
CUDA_PATH_13_0 := /usr/local/cuda-13.0

# Check which CUDA version exists and set CUDA_PATH accordingly
ifneq ($(wildcard $(CUDA_PATH_13_0)/bin/nvcc),)
    CUDA_PATH ?= $(CUDA_PATH_13_0)
    CUDA_VERSION := 13.0
else ifneq ($(wildcard $(CUDA_PATH_12_8)/bin/nvcc),)
    CUDA_PATH ?= $(CUDA_PATH_12_8)
    CUDA_VERSION := 12.8
else
    $(error No CUDA installation found. Please install CUDA 12.8 or 13.0)
endif

NVCC := $(CUDA_PATH)/bin/nvcc

INCLUDES := -I$(CUDA_PATH)/include -I.
LIBRARIES := -L$(CUDA_PATH)/lib64

# Compiler flags - no architecture specification to allow auto-detection
NVCC_FLAGS := -O2
LDFLAGS := -lcudart

# Add flag for CUDA 12.8 with newer GCC
ifeq ($(CUDA_VERSION),12.8)
    NVCC_FLAGS += -allow-unsupported-compiler
endif

# Pass CUDA version as a define
NVCC_FLAGS += -DCUDA_VERSION_MAJOR=$(shell echo $(CUDA_VERSION) | cut -d. -f1)

TARGETS = vectorAdd matrixMul bandwidthTest deviceQuery

all: 
	@echo "Building with CUDA $(CUDA_VERSION) from $(CUDA_PATH)"
	@$(MAKE) $(TARGETS)

vectorAdd: vectorAdd.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

matrixMul: matrixMul.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

bandwidthTest: bandwidthTest.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

deviceQuery: deviceQuery.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.cubin *.ptx *.fatbin

run: all
	@echo "Running vectorAdd..."
	./vectorAdd
	@echo "\nRunning matrixMul..."
	./matrixMul
	@echo "\nRunning bandwidthTest..."
	./bandwidthTest
	@echo "\nRunning deviceQuery..."
	./deviceQuery

# Allow user to specify CUDA version explicitly
cuda12.8:
	@$(MAKE) CUDA_PATH=$(CUDA_PATH_12_8) CUDA_VERSION=12.8 all

cuda13.0:
	@$(MAKE) CUDA_PATH=$(CUDA_PATH_13_0) CUDA_VERSION=13.0 all

.PHONY: all clean run cuda12.8 cuda13.0