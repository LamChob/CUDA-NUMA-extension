#include "global.h"
#include "migration.h"
#include "numa-interface.h"

#include <iostream>
#include <numeric>
#include <vector>
#include <fstream>
#include <omp.h>

constexpr int block_size {1024};


struct benchmark_config {
    int n_it, n_t;
    uint64_t size;
    bool prefetch, affinity, check; 
};

struct benchmark_timers {
    double alloc,
        cpu_load1,
        gpu_load1,
        h2d1,
        d2h1,
        total1,
        cpu_load2,
        gpu_load2,
        h2d2,
        d2h2,
        total2,
        h2h;
};

__global__ void gpu_load(float* d, int n) {
    for(int inx = 0; inx < n; inx+= blockDim.x) 
        d[inx] += 1.0f;
}

__host__ void cpu_load(float* d, int n, int node) {
    int n_threads = get_num_cores();
    omp_set_num_threads(n_threads); 
    #pragma omp parallel
    {
        node_pin(node);
        #pragma omp for
        for (int i = 0; i < n; i++) {
            d[i] += 1.0f;
        }
    }
}

void print_csv(std::string sep, benchmark_timers& timer, benchmark_config& conf) {
    // size | n. o. iterations | total[ms] | allocation[ms] | h2d[ns] | gpu[ms] | cpu[ms] | d2h[ms] | h2d[gbs] | d2h[hbs]
    double val;
    val = conf.affinity == true ? timer.h2h : 0.0;
    std::cout << conf.size << sep; 
    std::cout << conf.n_it << sep; 
    std::cout << conf.n_t << sep; 
    std::cout << timer.alloc << sep;
    std::cout << val << sep;
    
    std::cout << timer.total1 << sep;
//    std::cout << timer.h2d1 << sep;
//    std::cout << timer.gpu_load1 << sep;
//    std::cout << timer.cpu_load1 << sep;
//    std::cout << timer.d2h1 << sep;

    std::cout << timer.total2 << sep;
//    std::cout << timer.h2d2 << sep;
//    std::cout << timer.gpu_load2 << sep;
//    std::cout << timer.cpu_load2 << sep;
//    std::cout << timer.d2h2 << sep;
    
    val = conf.prefetch == true ? conf.size/timer.h2d1 : 0.0; std::cout << val << sep;
    val = conf.prefetch == true ? conf.size/timer.d2h1 : 0.0; std::cout << val << sep;
    val = conf.prefetch == true ? conf.size/timer.h2d2 : 0.0; std::cout << val << sep;
    val = conf.prefetch == true ? conf.size/timer.d2h2 : 0.0; std::cout << val << sep;
    val = conf.affinity == true ? conf.size/(timer.h2h * 1000000) : 0.0; std::cout << val << sep;

    std::cout << std::endl;
}

bool check_result(float* d, int nit) {
    return true;
}

double run_gpu(float* d, int n_it, int size) {
    Timer t_local;
    t_local.start();
    for(int i = 0; i < n_it; i++) {
        gpu_load<<<1, block_size>>>(d, size/sizeof(float));  
        cudaDeviceSynchronize();
    }
    t_local.stop();
    return t_local.elapsed<ms>();
}

double run_cpu(float* d, int n_it, int size, int node) {
    Timer t_local;
    t_local.start();
    for(int i = 0; i < n_it; i++) {
        cpu_load(d, size/sizeof(float), node);  
    }
    t_local.stop();
    return t_local.elapsed<ms>();
}

void app_run(float* d, benchmark_config& conf, int gpu, int node, benchmark_timers& timer) {
    numaMemPrefetchAsync(d, conf.size, node);
    cudaDeviceSynchronize();
    cudaSetDevice(gpu);
    Timer t_local, t_total; 
    //std::cout << get_pos(d) << std::endl;
//========== 0 - 0 ==============================
    t_total.start();
    for(int i = 0; i < conf.n_t; i++) {
        if (conf.prefetch) {
            t_local.start();
            numaMemPrefetchAsync(d, conf.size, gpu);
            cudaDeviceSynchronize();
            t_local.stop();
            timer.h2d1 = t_local.elapsed<ns>();
        }
        
        timer.gpu_load1 += run_gpu(d, conf.n_it, conf.size);

        if (conf.prefetch) {
            t_local.start();
            numaMemPrefetchAsync(d, conf.size, node);
            cudaDeviceSynchronize();
            t_local.stop();
            timer.d2h1 = t_local.elapsed<ns>();
        }

        timer.cpu_load1 += run_cpu(d, conf.n_it, conf.size,  node);
    }
    t_total.stop();
    timer.total1 = t_total.elapsed<ms>();

//========== 1 - x ==============================
    t_total.start();
    gpu += 1;
    cudaSetDevice(gpu);
    if (conf.affinity) {
        numaGetAffinity(gpu, &node);
        t_local.start();
        numaMemPrefetchAsync(d, conf.size, node);
        cudaDeviceSynchronize();
        t_local.stop();
        timer.h2h = t_local.elapsed<ms>();
    }
    //std::cout << get_pos(d) << std::endl;

    for(int i = 0; i < conf.n_t; i++) {

        if (conf.prefetch) {
            t_local.start();
            numaMemPrefetchAsync(d, conf.size, gpu);
            cudaDeviceSynchronize();
            t_local.stop();
            timer.h2d2 = t_local.elapsed<ns>();
        }

        timer.gpu_load2 += run_gpu(d, conf.n_it, conf.size);

        if (conf.prefetch) {
            t_local.start();
            numaMemPrefetchAsync(d, conf.size, node);
            cudaDeviceSynchronize();
            t_local.stop();
            timer.d2h2 = t_local.elapsed<ns>();
        }

        timer.cpu_load2 += run_cpu(d, conf.n_it, conf.size, node);
    }
    t_total.stop();
    timer.total2 = t_total.elapsed<ms>();
}

int main(int argc, char **argv) {
    benchmark_config conf; 
    cmdline::parser parser;
    parser.add<uint64_t>("size", 's', "data size, in MByte", true);
    parser.add<int32_t>("nit", 'i', "number of kernel iterations", true);
    parser.add<int32_t>("nt", 't', "number of timesteps", true);
    parser.add<bool>("check", 'c', "check results", false, false);
    parser.add<bool>("affinity", 'a', "force node/gpu affinity", false, true);
    parser.add<bool>("prefetch", 'p', "prefetch or page-faults", false, true);

    parser.parse_check(argc, argv);

    conf.n_it      = parser.get<int32_t>("nit");
    conf.n_t       = parser.get<int32_t>("nt");
    conf.size      = parser.get<uint64_t>("size") * 1024 * 1024;
    conf.check     = parser.get<bool>("check");
    conf.affinity  = parser.get<bool>("affinity");
    conf.prefetch  = parser.get<bool>("prefetch");

    float *data;
    int work_node, work_gpu;
    cudaError_t c_err;
    benchmark_timers timers;
    Timer t;

    work_gpu = 0;
    numaGetAffinity(work_gpu, &work_node);

    t.start();
    c_err = numaMallocManaged((void**)&data, conf.size, cudaMemAttachGlobal, work_node);
    if ( c_err != cudaSuccess ) {
        std::cout << "Allocation Failed: " << cudaGetLastError() << std::endl;
    }
    t.stop();
    timers.alloc = t.elapsed<ms>();

    app_run(data, conf, work_gpu, work_node, timers);    
    print_csv(" ", timers, conf);
    
    cudaFree(data);
}
