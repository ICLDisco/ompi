CC			= gcc
NVCC		= nvcc
ARCH		= ar
ARCHFLAGS	= cr
RANLIB		= ranlib
STLIB		?= opal_datatype_cuda.a
DYLIB		?= opal_datatype_cuda.so
CFLAGS		= -g -G -O0 
EXTLIB		= -L/home/wwu12/ompi/ompi-gpu/opal/datatype/.libs -ldatatype -L/usr/lib64 -lcuda
INC			=

SRC	:= \
    opal_datatype_cuda.cu \
    opal_datatype_pack_cuda_kernel.cu \
    opal_datatype_pack_cuda_wrapper.cu \
	opal_datatype_unpack_cuda_kernel.cu \
	opal_datatype_unpack_cuda_wrapper.cu \
	
OBJ := $(SRC:.cu=.o)

.PHONY: all clean cleanall

all: $(STLIB) $(DYLIB)

$(STLIB): $(OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(OBJ) 
	$(RANLIB) $@
	
$(DYLIB): $(OBJ)
	$(NVCC) $(CFLAGS) $(EXTLIB) -shared --compiler-options '-fPIC' -o $(DYLIB) $(OBJ)
	
%.o: %.cu
	$(NVCC) $(CFLAGS) $(EXTLIB) -gencode arch=compute_35,code=sm_35 $(INC) -c --compiler-options '-fPIC' $< -o $@ 

clean:
	rm -f *.o

cleanall: clean
	rm -f $(STLIB)
	rm -f $(DYLIB)
