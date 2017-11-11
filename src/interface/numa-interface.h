#include <cuda_runtime.h>
#include <migration.h>

#include <map>
#include <vector>

#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#define gettid() syscall(SYS_gettid)

#include <dirent.h>

struct migration_args {
    int     node;
    void*   data_src;
    void*   data_dst;
    size_t  size;
};

static std::map<void*,std::vector<void*>> _ndir; // directory to remember all pointers


__host__ cudaError_t numaGetAffinity(int device, int* node) {
    static int ndevices = 0; 
    if (ndevices == 0) cudaGetDeviceCount(&ndevices);
    *node = get_cuda_affinity(device);
    *node += ndevices;
    return cudaSuccess;
}

__host__ void _migration_callback(cudaStream_t stream, cudaError_t status, void *vargs) {
    migration_args *args = (migration_args*) vargs;
    int n_threads = get_num_cores();
    std::vector<std::thread> copy_threads(n_threads); 
    int nt = 0;
    for(auto& t : copy_threads) {
        t = std::thread([=](){
            char* dst = (char*)args->data_dst; 
            char* src = (char*)args->data_src; 
            unsigned int copy_slice  = (args->size/n_threads);
            size_t size = copy_slice;

            dst += nt * copy_slice;
            src += nt * copy_slice;
            memcpy((void*) dst, (void*) src, size);
        });
        ++nt;
    }
    for(auto& t : copy_threads) {
        t.join();
    }
    memcpy(args->data_dst, args->data_src, args->size);
    delete args;
}

__host__ cudaError_t numaGetDeviceCount(int *nd) {
    cudaGetDeviceCount(nd);
    *nd += get_num_nodes();
    return cudaSuccess;
}

template<typename T> __host__ cudaError_t numaMemPrefetchAsync(T*& dev_ptr, size_t size, int dstDevice, cudaStream_t stream = 0) {
    cudaError_t ret; 
    int n_devices; 
    cudaPointerAttributes attr;

    cudaPointerGetAttributes(&attr, (const void*) dev_ptr);
    cudaGetDeviceCount(&n_devices);

    if (dstDevice < n_devices) { // target is CUDA Device
        ret = cudaMemPrefetchAsync(dev_ptr, size, dstDevice, stream);
    } else if (dstDevice >= n_devices){ // target is numa node
        int node = dstDevice - n_devices;
        void* dst_ptr; 
        dst_ptr = _ndir[dev_ptr][node];
        _ndir[dst_ptr] = _ndir[dev_ptr]; // store other nodes

        if (dst_ptr != dev_ptr) { // check if we need to copy back, or also migrate
            migration_args *arg = new migration_args; // freed in callback
            arg->data_src = dev_ptr;
            arg->data_dst = dst_ptr;
            arg->size     = size;
            arg->node     = dstDevice - n_devices;

            ret = cudaMemPrefetchAsync(dev_ptr, size, cudaCpuDeviceId, stream); // does not improve performance, but reduces page faults
            ret = cudaStreamAddCallback(stream, _migration_callback, (void*) arg, 0);

            if (ret != cudaSuccess) delete arg;
            else dev_ptr = (T*) dst_ptr;
        } else {
            ret = cudaMemPrefetchAsync(dst_ptr, size, cudaCpuDeviceId, stream);
        }
    }    
    return ret;
}

__host__ cudaError_t numaMallocManaged(void** dev_ptr, size_t size, unsigned int flags = cudaMemAttachGlobal, int node = -1) { 
    cudaError_t ret;
    cudaFree(0); // init context
    std::vector<int> cuda_context_tid(0);

    static int ndevices = 0; 
    if (ndevices == 0) cudaGetDeviceCount(&ndevices);
    node -= ndevices;
    node = node < 0 ? ndevices : node;


    // get all potential context thread-ids and bind also bind to the node
    if(DIR* dir = opendir("/proc/self/task")) {
    while (dirent* entry = readdir(dir))
        if (entry->d_name[0] != '.') {
            cuda_context_tid.push_back(atoi(entry->d_name));
            int r = node_pin(node, atoi(entry->d_name));
            if ( r != 0 ) { std::cout << "e: " << strerror(errno) << std::endl; }
        }
        closedir(dir);
    }

    // allocate on target node
    ret = cudaMallocManaged(dev_ptr, size, flags);
    if (ret != cudaSuccess) return ret;
    
    _ndir[*dev_ptr].resize(get_num_nodes()); 
    _ndir[*dev_ptr][node] = *dev_ptr;
    for(int inx = 0; inx < size/sizeof(float); inx += 4096/sizeof(float)) {
        float* t = (float*)*dev_ptr;
        t[inx] = 0.0f;
    }

    // allocate on the rest of the nodes
    for (int i = 0; i < get_num_nodes(); i++ ) {
        if (i == node) continue;
        node = 1;
        void* tmp; 
        for ( auto tid : cuda_context_tid ) {
            int r = node_pin(node, tid);
            if ( r != 0 ) { std::cout << "e: " << strerror(errno) << std::endl; }
        }
        ret = cudaMallocManaged(&tmp, size, flags);
        _ndir[*dev_ptr][i] = tmp;
        for(int inx = 0; inx < size/sizeof(float); inx += 4096/sizeof(float)) {
            float* t = (float*)tmp;
            t[inx] = 0.0f;
        }
        if (ret != cudaSuccess) return ret;
    }

    // remove bindings for all threads
    for ( auto tid : cuda_context_tid ) {
        int r = node_reset(tid);
        if ( r != 0 ) { std::cout << "e: " << strerror(errno) << std::endl; }
    }
    return ret;
}

__host__ cudaError_t numaFree(void* ptr) {
    for(auto& e : _ndir[ptr]) {
        if (e != NULL) cudaFree(e); 
        _ndir.erase(e);
    } 
    _ndir.erase(ptr);
    return cudaSuccess;
}
