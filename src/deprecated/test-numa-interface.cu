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

    float* data; 
    int size = atoi(argv[1]);

    //cout << "Cuda Devices: " << n_devices << " ";
//    cout << "NUMA Devices: " << _get_num_nodes() << endl;
    //numaGetDeviceCount(&n_devices);
    //cout << "Total Devices: "<< n_devices << endl;



    dim3 block(1024); 
    dim3 grid(1);
    cout << size <<  " " << size*sizeof(float);
    if ( numaMallocManaged((void**)&data, (size_t)size*sizeof(float), cudaMemAttachGlobal, 0) != cudaSuccess){ 
        std::cout << "Malloc Fail: " << cudaGetLastError() << std::endl; 
        return 0;
    } 
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) cout << "ERROR1: " << e <<endl;

    cout << " " <<  get_pos(data);
    numaMemPrefetchAsync(data, size*sizeof(float),0);
    e = cudaGetLastError();
    if (e != cudaSuccess) cout << "ERROR2: " << e <<endl;
    doSomethingKernel<<<grid, block>>>(data, size);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) cout << "ERROR3: " << e <<endl;
    //cout << "Result: " << data[3] << endl;

    auto t1 = system_clock::now();
    e = numaMemPrefetchAsync(data, size*sizeof(float), 2);
    if ( e != cudaSuccess) {
        cout << "prefetch Fail: " << cudaGetLastError() << endl;
    } //D2H
    cudaDeviceSynchronize();
    auto t2 = system_clock::now();
    double mt = duration_cast<nanoseconds>(t2-t1).count();
    cout <<";" << (size*sizeof(float))/mt << " " << get_pos(data);
    
    
    t1 = system_clock::now();
    e = numaMemPrefetchAsync(data, size*sizeof(float), 3);
    if ( e != cudaSuccess) {
        cout << "prefetch Fail: " << cudaGetLastError() << endl;
    } //D2H
    cudaDeviceSynchronize();
    t2 = system_clock::now();
    mt = duration_cast<nanoseconds>(t2-t1).count();
    cout <<";" << (size*sizeof(float))/mt << " " << get_pos(data);

    numaMemPrefetchAsync(data, size*sizeof(float),0);
    doSomethingKernel<<<grid, block>>>(data, size);
    cudaDeviceSynchronize();
    t1 = system_clock::now();
    e = numaMemPrefetchAsync(data, size*sizeof(float), 3);
    if ( e != cudaSuccess) {
        cout << "prefetch Fail: " << cudaGetLastError() << endl;
    } //D2H
    cudaDeviceSynchronize();
    t2 = system_clock::now();
    mt = duration_cast<nanoseconds>(t2-t1).count();
    cout <<";" << (size*sizeof(float))/mt << " " << get_pos(data) <<  ";" << data[0] << endl;

    numaFree(data);
}
