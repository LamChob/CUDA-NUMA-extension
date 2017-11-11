/*
 * Author: Dennis Sebastian Rieber, <rieber@stud.uni-heidelberg.de>
 * Generate utility for NUMA Migration of data
 */
#pragma once

#include <thread>
#include <vector>
#include <utility>

#include <iostream>
#include <numaif.h>
#include <hwloc.h>
#include <hwloc/cudart.h>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#define gettid() syscall(SYS_gettid)

//#include <migration.h>

#ifndef USE_PAGE_MIGRATION
    #define FIRST_TOUCH
#else 
    #define PAGE_MIGRATION
#endif

typedef unsigned int node_t;

struct membind_pol {
    hwloc_membind_policy_t  pol;
    hwloc_membind_flags_t   flag;
    hwloc_nodeset_t         ns;
};

// this is a "global" structure that describes the system.
// creation is expensive and only required once since the
// system configuration is not expected changed during execution

static bool initialized {false};
static hwloc_topology_t _top;
static std::vector<hwloc_nodeset_t> _nodes; 

void _init() {
    hwloc_topology_init(&_top);
    hwloc_topology_load(_top);
    _nodes.resize(hwloc_get_nbobjs_by_type(_top, HWLOC_OBJ_NUMANODE));
    for(int i = 0; i < _nodes.size(); i++ ) {
        hwloc_obj_t o = hwloc_get_obj_by_type(_top, HWLOC_OBJ_NUMANODE, i);
        _nodes[i] = o->nodeset; 
    }
    initialized = true;
}



void _pin_to_node(node_t node) {
    hwloc_nodeset_t ns;    
        char *str;

 //   assert(node >= _nodes.size()); // trying to move to non-existant node

    //obj = hwloc_get_obj_by_type(_top, HWLOC_OBJ_NUMANODE, node);
    ns  = _nodes[node];

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_cpuset_from_nodeset(_top, cpuset, ns);

    hwloc_bitmap_singlify(cpuset);
    hwloc_bitmap_asprintf(&str, cpuset); 
    std::cout << str << std::endl;                       

    if (hwloc_set_cpubind(_top, cpuset, HWLOC_CPUBIND_PROCESS)) { // TODO error handling
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, cpuset);
        std::cout << str << std::endl;                       
        free(str);
    }
    free(str);
    float* something = new float;
    //_print_pos(something);

    hwloc_bitmap_free(cpuset);
}

int _migrate_ft(void** d, size_t size, node_t node) {
    _pin_to_node(node);    
    char *old;
    char *nd = new char[size]; //TODO use different way to allocate memory
    memcpy(nd, *d, size);
    old = (char*) *d;
    *d =(void*) nd;
    delete[] old;
    return 0;
}

int _migrate_pm(void** d, size_t size, node_t node) {
    hwloc_nodeset_t ns;    

  //  assert(node >= _nodes.size()); // trying to move to non-existant node
    //obj = hwloc_get_obj_by_type(_top, HWLOC_OBJ_NUMANODE, node);
    ns = _nodes[node];

    return hwloc_set_area_membind_nodeset(_top, *d, size, ns, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_MIGRATE);
}

int set_membind_to_node(node_t node, membind_pol *reset, long ptid=0) {
    if (!initialized) _init();
    if (ptid == 0) ptid = syscall(SYS_gettid);
    hwloc_nodeset_t ns = hwloc_bitmap_alloc();
    reset->ns = hwloc_bitmap_alloc();
    hwloc_get_membind_nodeset(_top, ns, &(reset->pol), 0);
    reset->ns = hwloc_bitmap_dup(ns);

	hwloc_bitmap_clr_range(ns, 0, -1);hwloc_bitmap_set(ns, node);
    int ret = hwloc_set_membind_nodeset(_top, ns, HWLOC_MEMBIND_BIND, 0 | HWLOC_MEMBIND_STRICT);
    hwloc_bitmap_free(ns);
    return ret;
}
int set_membind_policy(membind_pol *reset, long ptid = 0) {
    if (ptid == 0) ptid = syscall(SYS_gettid);;
    int ret = hwloc_set_membind_nodeset(_top, reset->ns, reset->pol, 0 | HWLOC_MEMBIND_STRICT);
    hwloc_bitmap_free(reset->ns);
    return ret;
}

int alloc_to_node(void** p, size_t n, node_t node) {
    hwloc_nodeset_t ns = hwloc_bitmap_alloc();
	hwloc_bitmap_clr_range(ns, 0, -1);hwloc_bitmap_set(ns, node);
    *p = hwloc_alloc_membind_nodeset(_top, n, ns, HWLOC_MEMBIND_BIND, 0);
    if (p == NULL ) std::cout << "ERROR: " << errno << std::endl;
    hwloc_bitmap_free(ns);
    return 0;
}

int get_num_cores(int node = 0) {
    if (!initialized) _init();
    static int n_cores {0};
    if ( n_cores == 0 ) {
        hwloc_nodeset_t ns  = _nodes[node];
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
        hwloc_cpuset_from_nodeset(_top, cpuset, ns);
        n_cores = hwloc_bitmap_weight(cpuset);
        hwloc_bitmap_free(cpuset);
    }
    return n_cores;
}

__inline__ int get_num_nodes() {
    if (!initialized) _init();
    return _nodes.size();
}

int node_pin(int core, int pid = 0) {
    cpu_set_t cp;
    CPU_ZERO(&cp);
    CPU_SET(core*4, &cp);
    pid = pid == 0 ? gettid() : pid;
    return sched_setaffinity(pid, sizeof(cpu_set_t), &cp);
}

int node_reset(int pid) {
    cpu_set_t cp;
    CPU_ZERO(&cp);
    for(int i = 0; i < get_num_nodes() * get_num_cores(); i++) CPU_SET(i, &cp); // TODO
    return sched_setaffinity(pid, sizeof(cpu_set_t), &cp);
}


int get_pos(void* d) {
    if (!initialized) _init();
    unsigned long count = 1;
    int flags {0};
    int* status = new int[count];
    void** pages = new void*[count];

    pages[0] = d;
    move_pages(0, count, (void**) pages,NULL, status, flags);
   // std::cout << " _pos: " << status[0] << std::endl;
    
    int ret = status[0];
    delete[] status;
    delete[] pages;
    return ret;
}

int get_cuda_affinity(int device) {
    if (!initialized) _init();
    int affinity_node;
    hwloc_cpuset_t cs;
    hwloc_nodeset_t ns;

    ns = hwloc_bitmap_alloc();
    cs = hwloc_bitmap_alloc();

    hwloc_cudart_get_device_cpuset(_top, device, cs);
    hwloc_cpuset_to_nodeset(_top, cs, ns);
    hwloc_bitmap_singlify(ns);
    
    affinity_node = hwloc_bitmap_first(ns);

    hwloc_bitmap_free(ns);
    hwloc_bitmap_free(cs);

    return affinity_node;
}

int migrate(void** d, size_t size, node_t node) {
    if (!initialized) _init();
#ifdef FIRST_TOUCH
    auto t = std::thread(_migrate_ft, d, size, node);
    t.join();
    return 0;
#endif
#ifdef PAGE_MIGRATION
    return _migrate_pm(d, size, node);
#endif
}
