all:
	nvcc -std=c++11 -lhwloc  managed-mem-hwloc.cu -arch=sm_61 -o managed-test -lnuma -O3
	g++ -std=c++11 -lhwloc  hwloc-corepin.cpp -o ftt -lpthread -lnuma -O0
