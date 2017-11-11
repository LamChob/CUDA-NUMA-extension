#include <hwloc.h>
#include <iostream>
#include <thread>
#include <numaif.h>
#include <chrono>

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

    pages[0] = d;
    move_pages(0, count, (void**) pages,NULL, status, flags);
    cout << " pos: " << status[0] << endl;

    delete[] status;
    delete[] pages;
}

void hwloc_bind_thread(int tn) {
    char* string = new char[128];
/*
    // bind thread to cpu
    int depth = hwloc_get_type_or_below_depth(top, HWLOC_OBJ_CORE);
    hwloc_obj_t obj;
    obj = hwloc_get_obj_by_depth(top, depth, tn);

    hwloc_cpuset_t last, current;
    last = hwloc_bitmap_alloc();
    current = hwloc_bitmap_alloc();
    if (obj) {
        hwloc_get_last_cpu_location(top, last, HWLOC_CPUBIND_THREAD);
	    hwloc_bitmap_asprintf(&string, last);
        cout << "Last: " << string << endl;

        // Get a copy of its cpuset that we may modify.
        hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->cpuset);
        // Get only one logical processor (in case the core is SMT/hyper-threaded).
        hwloc_bitmap_singlify(cpuset);
        // And try to bind ourself there. 
        if (hwloc_set_cpubind(top, cpuset, HWLOC_CPUBIND_THREAD)) {
            char *str;
            int error = errno;
            hwloc_bitmap_asprintf(&str, obj->cpuset);
            printf("Couldn’t bind to cpuset %s: %s\n", str, strerror(error));
            free(str);
        }
        // Free our cpuset copy
        hwloc_bitmap_free(cpuset);

        hwloc_get_last_cpu_location(top, current, HWLOC_CPUBIND_THREAD);
	    hwloc_bitmap_asprintf(&string, current);
        cout << "Current: " << string << endl;
        //cout << "Compare: " << hwloc_bitmap_compare(last, current) << endl;
    
    } else { cout << "No OBJ!" << endl; }
    delete[] string; */


    hwloc_cpuset_t last, current, ns;
    last = hwloc_bitmap_alloc();
    current = hwloc_bitmap_alloc();
    ns = hwloc_bitmap_alloc();
    hwloc_get_last_cpu_location(top, last, HWLOC_CPUBIND_THREAD);
    hwloc_cpuset_to_nodeset(top, last, ns);
    hwloc_bitmap_asprintf(&string, ns);
    cout << "Last: " << string << endl;

    hwloc_obj_t obj;     
    int n_nodes = hwloc_get_nbobjs_by_type(top, HWLOC_OBJ_NUMANODE); 
    //assert(node >= n_nodes); // trying to move to non-existant node 
    obj = hwloc_get_obj_by_type(top, HWLOC_OBJ_NUMANODE, tn); 
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc(); 
    hwloc_cpuset_from_nodeset(top, cpuset, obj->nodeset); 
    if (hwloc_set_cpubind(top, cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT)) { 
        char *str;                                                   
        int error = errno; 
        hwloc_bitmap_asprintf(&str, obj->cpuset); 
        cout << str << endl;                       
        free(str);                                                         
    }               
    cout << "error: " << strerror(errno) << endl;
    hwloc_get_last_cpu_location(top, current, HWLOC_CPUBIND_THREAD);
    hwloc_cpuset_to_nodeset(top, current, ns);
    hwloc_bitmap_asprintf(&string, ns);
    cout << "Current: " << string << endl;
    hwloc_bitmap_free(cpuset); 
    hwloc_bitmap_free(current); 
    hwloc_bitmap_free(last); 
}

void hwloc_ft(float** d, int n){
    hwloc_bind_thread(0);    
    // allocate new data
    float *nd = new float[n];
    for(int i = 0; i < _size; i++ ) nd[i] = 2.0f;
    memcpy(nd, *d, n*sizeof(float));
    std::swap(*d, nd);
    delete nd;
}
void core_pin(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET( core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
void ft(float** d, int n) {
    hwloc_bind_thread(0);
    *d = (float*)malloc(sizeof(float)*n);//new float[n];
    memset(*d, 0, n*sizeof(float));
}

int main(int argc, char* argv[]) {
	assert(!(argc < 2));
	char *s;

	hwloc_topology_init(&top);
	hwloc_topology_load(top);

    //hwloc_bind_thread(0);    
	_size = atoi(argv[1]);

    auto t = thread(ft, &_data, _size);
    t.join();
    print_pos(_data);
    hwloc_bind_thread(0);
    hwloc_bind_thread(1);
    float a = 1;
    print_pos(&a);

/*
    _data = new float[_size];

    for(int i = 0; i < _size; i++ ) _data[i] = 1.0f;
    cout << "Original Pointer: " << _data << " content is: " << _data[0] << endl;
    print_pos(_data);
    
    auto t1  = system_clock::now();
    thread t(hwloc_ft, &_data, _size); 
    t.join();
    auto t2  = system_clock::now();
    double mt = (double) duration_cast<nanoseconds>(t2-t1).count();
    cout << "Bandwidth: " << (_size * sizeof(float)) / mt << " GByte/s" << endl;
    

    cout << "New Pointer: " << _data << " content is: " << _data[0] << endl;
    print_pos(_data);
    t1 = system_clock::now();
    hwloc_bind_thread(false);
    t2 = system_clock::now();
    mt = (double) duration_cast<nanoseconds>(t2-t1).count();
    cout << "Thread Migration: " << mt << " ns" << endl;

*/
    delete _data;
}
