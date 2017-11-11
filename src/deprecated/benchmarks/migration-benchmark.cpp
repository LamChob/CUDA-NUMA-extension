#include <numavec.h>
#include <iostream>
#include <chrono>
#include <numaif.h>
#include <assert.h>
#include <hwloc.h>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

// return bandwidth
double move_data(numavec<float>& a, const int dst_node) {
	auto t1 = system_clock::now();
	switch(dst_node) {
		case 1 : a.migrate({2});break;
		case 2 : a.migrate({2,3});break;
		case 3 : a.migrate({2,3,4});break;
		case 4 : a.migrate({2,3,4,5});break;
		case 5 : a.migrate({2,3,4,5,6});break;
		case 6 : a.migrate({2,3,4,5,6,7});break;
		case 7 : a.migrate({1,2,3,4,5,6,7});break;
		case 8 : a.migrate({0,1,2,3,4,5,6,7});break;
	}
	auto t2 = system_clock::now();

	auto time = static_cast<double>(duration_cast<nanoseconds>(t2-t1).count());
	auto size = static_cast<double>(a.size()*sizeof(float));

	return size/time;
}

int main( int argc, char* argv[] ) {
	if ( argc < 3 ) exit(255);
	const int sz = atoi(argv[1]) * 1024 * 1024;
	const int num_nodes = atoi(argv[2]);

	hwloc_topology_t top;
	hwloc_obj_t obj;
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
	hwloc_membind_policy_t policy;
	int err;

	hwloc_topology_init(&top);
	hwloc_topology_load(top);

	numavec<float> a(sz);
	for(int i = 0; i < sz; i++)
		a[i] = 1.0f;

	a.migrate(0); // set node to source

  	/*err = hwloc_get_area_membind(top, &a[0], sz, set, &policy, HWLOC_MEMBIND_BYNODESET);
  	hwloc_bitmap_asprintf(&s, set);
    cout << "Membind Nodeset: " << s << endl;*/
		
	// pin this thread to the src_node
	obj = hwloc_get_obj_by_type(top, HWLOC_OBJ_NUMANODE, 0);	
	hwloc_set_cpubind(top, obj->cpuset, HWLOC_CPUBIND_THREAD);
	
	auto bw = move_data(a, num_nodes);	
	if( bw <= 0.0 ) {
		cout << "Migration Error: " << bw << endl;
		exit(255);
	}
	cout << sz*sizeof(float) << "  " << num_nodes << " " << bw << endl;
  	/*err = hwloc_get_area_membind(top, &a[0], sz, set, &policy, HWLOC_MEMBIND_BYNODESET);
  	hwloc_bitmap_asprintf(&s, set);
    cout << "Membind Nodeset: " << s << endl;*/
	
}
