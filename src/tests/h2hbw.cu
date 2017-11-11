#include <cuda_runtime.h>
#include <numa-interface.h>
//#include <migration.h>

#include <iostream>
#include <chrono>

#include <numaif.h>
#include <hwloc.h>

using namespace std;
using namespace chrono;

__global__ void doSomethingKernel(float *in, int sz) {
    for(int inx = 0; inx < sz; inx+= 1024) 
        in[inx + threadIdx.x] += 5.0f;
}


int main(int argc, char* argv[]) {
	assert(!(argc < 2));

    int n_devices; 
    cudaGetDeviceCount(&n_devices);
    cudaError_t e;
    float* data1; 
    unsigned long size = atoi(argv[1]);
    size *= 1024*1024;

    int gpu = 0;
    int node;  
    numaGetAffinity(gpu, &node);

    dim3 block(1024); 
    dim3 grid(1);

    if ( numaMallocManaged((void**)&data1, (size_t)size, cudaMemAttachGlobal, 0) != cudaSuccess){  // allocate on 0
        std::cout << "Malloc Fail: " << cudaGetLastError() << std::endl; 
        return 0;
    } 

    // Fetch data to host
    e = numaMemPrefetchAsync(data1, size, node);
    if ( e != cudaSuccess) {
        cout << "prefetch Fail: " << cudaGetLastError() << endl;
    } //D2H
    cudaDeviceSynchronize();
    unsigned long h2h_time = 0;

    for(int i = 0; i < 10; i++) {
        // Benchmark H2H
        auto t1 = system_clock::now();
        e = numaMemPrefetchAsync(data1, size, 2);
        if ( e != cudaSuccess) {
            cout << "prefetch Fail: " << cudaGetLastError() << endl;
        } //H2H
        cudaDeviceSynchronize();
        auto t2 = system_clock::now();
        h2h_time += duration_cast<nanoseconds>(t2-t1).count();

        t1 = system_clock::now();
        e = numaMemPrefetchAsync(data1, size, 3);
        if ( e != cudaSuccess) {
            cout << "prefetch Fail: " << cudaGetLastError() << endl;
        } //H2H
        cudaDeviceSynchronize();
        t2 = system_clock::now();
        h2h_time += duration_cast<nanoseconds>(t2-t1).count();

    }
    std::cout << size/(1024*1024) << " " << size/((double)h2h_time/20) <<std::flush;

    e = numaMemPrefetchAsync(data1, size, 3);
    if ( e != cudaSuccess) {
        cout << "prefetch Fail: " << cudaGetLastError() << endl;
    }

    // measure node-to-device
    h2h_time = 0;
    for(int i = 0; i < 10; i++) {
        // Benchmark H2H
        auto t1 = system_clock::now();
        e = numaMemPrefetchAsync(data1, size, 0);
        if ( e != cudaSuccess) {
            cout << "prefetch Fail: " << cudaGetLastError() << endl;
        } //H2H
        cudaDeviceSynchronize();
        auto t2 = system_clock::now();
        h2h_time += duration_cast<nanoseconds>(t2-t1).count();

        doSomethingKernel<<<grid, block>>>(data1, size/sizeof(float));

        e = numaMemPrefetchAsync(data1, size, 3);
        if ( e != cudaSuccess) {
            cout << "prefetch Fail: " << cudaGetLastError() << endl;
        } //H2H
        cudaDeviceSynchronize();

    }
    std::cout << " " << size/((double)h2h_time/10) << std::endl;

    return 0;
}
