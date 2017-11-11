#include <hwloc.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <numaif.h>

using namespace std;
using namespace chrono;
		
hwloc_topology_t top;
float* _data;
unsigned int _size;

void print_pos(float* d) {
	unsigned long count = 1;
    int flags {0};
    int* status = new int[count];
   	float** pages = new float*[count]; 
    status[0] = -1;

    pages[0] = d;
    int r = move_pages(0, count, (void**) pages,NULL, status, flags);
    cout << " pos: " << status[0] << endl;
    if(r != 0) cout << "pp error: " << strerror(errno) << endl;

    delete[] status;
    delete[] pages;
}

__global__ void doSomethingKernel(float *in) {
    int tix = threadIdx.x + blockIdx.x * gridDim.x;
    in[tix] += (float)tix;
}

void hwloc_migrate(int target) {
	hwloc_nodeset_t ns = hwloc_bitmap_alloc();
	hwloc_membind_policy_t* p = new hwloc_membind_policy_t;
	char* string;

	hwloc_get_area_membind_nodeset(top, _data, static_cast<size_t>(_size), ns, p, 0);
	int first = hwloc_bitmap_first(ns);
	hwloc_bitmap_asprintf(&string, ns);
	cout << "Current Nodeset: " << string << endl;
	hwloc_bitmap_clr_range(ns, 0, -1);

	hwloc_bitmap_set(ns, target);
	hwloc_bitmap_asprintf(&string, ns);
	cout << "Target Nodeset: " << string << endl;

    *p = HWLOC_MEMBIND_BIND;
	int ret = hwloc_set_area_membind_nodeset(top, _data, _size*sizeof(float), ns, *p, HWLOC_MEMBIND_MIGRATE);
    if(ret != 0) cout << "error: " << strerror(errno) << endl;
	hwloc_get_area_membind_nodeset(top, _data, _size*sizeof(float), ns, p, 0);
	hwloc_bitmap_asprintf(&string, ns);
	cout << "New Nodeset: " << string << endl;

	delete p;
}

void core_pin(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET( core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

int main(int argc, char* argv[]) {
	assert(!(argc < 2));
	_size = atoi(argv[1]);
    cudaError_t e;
    char * string;
    int ret;

    dim3 block(1024); 
    dim3 grid(_size/1024)  ; 
    cout << _size << " " << block.x << " " << grid.x << endl;

	hwloc_topology_init(&top);
	hwloc_topology_load(top);
    
    
	hwloc_nodeset_t ns = hwloc_bitmap_alloc();
	hwloc_bitmap_clr_range(ns, 0, -1);hwloc_bitmap_set(ns, 1);
	hwloc_bitmap_asprintf(&string, ns);
	cout << "Target Nodeset: " << string << endl;
    ret = hwloc_set_membind_nodeset(top, ns, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
    if(ret != 0) cout << "error: " << strerror(errno) << endl;

    cudaMallocManaged(&_data, _size * sizeof(float));    
    memset(_data, 0, _size*sizeof(float));
    print_pos(_data);
    
    
    doSomethingKernel<<<grid, block>>>(_data);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) cout << "ERROR2: " << e <<endl;
    cout << "Result: " << _data[3] << endl;

	hwloc_bitmap_clr_range(ns, 0, -1);hwloc_bitmap_set(ns, 0);
	hwloc_bitmap_asprintf(&string, ns);
	cout << "Target Nodeset: " << string << endl;
    ret = hwloc_set_membind_nodeset(top, ns, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
    if(ret != 0) cout << "error: " << strerror(errno) << endl;

    float *_data2;
    cudaMallocManaged(&_data2, _size * sizeof(float));    
    memcpy(_data2, _data, _size * sizeof(float));
    print_pos(_data2);
    _data = _data2;
//    swap(_data, _data2);
    
    doSomethingKernel<<<grid, block>>>(_data);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) cout << "ERROR2: " << e <<endl;
    cout << "Result: " << _data[3] << endl;

    cudaFree(&_data);
    cudaFree(&_data2);
    return 0;
}
