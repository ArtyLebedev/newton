CXX        = mpicxx
LD         = $(CXX)

#LIBS_PATH = -L/opt/mpich-3.1/lib
#LIBS =  -lmpich -lpthread
#INCLUDE_PATH = -I/opt/mpich-3.1/include -I/home/cuda/cuda-7.5/include
FLAGS = -Wall
TARGET = newtone_solver
OBIN = ntest


all:$(TARGET)

$(TARGET):
	#$(LD) $(INCLUDE_PATH) $(FLAGS) timer_impl.cu timer_incl.cu timer.cu main_2d.cpp mpi_computator.cu cuda_computator.cu seq_computator.cpp thread_computator.cu -o $(OBIN) $(LIBS_PATH) $(LIBS)
	$(LD) $(FLAGS) ./src/main.cpp ./src/ntn_comp.cpp -o $(OBIN) 
