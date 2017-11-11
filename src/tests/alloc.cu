#include <cuda_runtime.h>
#include <numa-interface.h>
#include <migration.h>

#include <iostream>
#include <chrono>

#include <numaif.h>
#include <hwloc.h>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
	assert(!(argc < 2));

    int n_devices; 
    cudaGetDeviceCount(&n_devices);
    float* cuda; 
    float* cuda_memset;
    float* numa; 
    unsigned long size = atoi(argv[1]);
    size *= 1024*1024;
    unsigned long allocation_numa, allocation_cuda, allocation_cuda_memset;
    allocation_numa = allocation_cuda = allocation_cuda_memset = 0;
    int iterations = 10;

    for (int i = 0; i < iterations; i++) {
        auto t1 = system_clock::now();
        if ( numaMallocManaged((void**)&numa, (size_t)size, cudaMemAttachGlobal, 0) != cudaSuccess){  // allocate on 0
            std::cout << "Malloc Fail: " << cudaGetLastError() << std::endl; 
            return 0;
        } 
        auto t2 = system_clock::now();
        allocation_numa += duration_cast<microseconds>(t2-t1).count();
        
        
        t1 = system_clock::now();
        if ( cudaMallocManaged((void**)&cuda, (size_t)size) != cudaSuccess){  // allocate on 0
            std::cout << "Malloc Fail: " << cudaGetLastError() << std::endl; 
            return 0;
        } 
        t2 = system_clock::now();
        allocation_cuda = duration_cast<microseconds>(t2-t1).count();

        t1 = system_clock::now();
        if ( cudaMallocManaged((void**)&cuda_memset, (size_t)size) != cudaSuccess){  // allocate on 0
            std::cout << "Malloc Fail: " << cudaGetLastError() << std::endl; 
            return 0;
        } 
        memset(cuda_memset, 0, size);
        t2 = system_clock::now();
        allocation_cuda_memset = duration_cast<microseconds>(t2-t1).count();
    numaFree(numa);
    cudaFree(cuda);
    cudaFree(cuda_memset);
    } 

    std::cout << size << " " << allocation_cuda/10.0 << " " << allocation_numa/10.0 << " " << allocation_cuda_memset/10.0 << std::endl;

}
