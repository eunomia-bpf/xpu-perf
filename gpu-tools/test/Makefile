CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

INCLUDES := -I$(CUDA_PATH)/include -I.
LIBRARIES := -L$(CUDA_PATH)/lib64

NVCC_FLAGS := -arch=sm_52 -O2
LDFLAGS := -lcudart

TARGETS = vectorAdd matrixMul bandwidthTest deviceQuery

all: $(TARGETS)

vectorAdd: vectorAdd.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

matrixMul: matrixMul.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

bandwidthTest: bandwidthTest.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

deviceQuery: deviceQuery.cu helper_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBRARIES) $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

run: all
	@echo "Running vectorAdd..."
	./vectorAdd
	@echo "\nRunning matrixMul..."
	./matrixMul
	@echo "\nRunning bandwidthTest..."
	./bandwidthTest
	@echo "\nRunning deviceQuery..."
	./deviceQuery

.PHONY: all clean run