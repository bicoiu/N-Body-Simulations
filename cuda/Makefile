NVCC ?= nvcc
SRC := nbody_cuda.cu
BIN := nbody_cuda

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) -O3 -std=c++14 -o $@ $<

clean:
	$(RM) $(BIN)

.PHONY: all clean
