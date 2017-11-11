#include <hwloc.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

hwloc_topology_t top;
float* _data;
int _size;

void hwloc_migrate() {
//	hwloc_nodeset_t ns = hwloc_bitmap_dup(hwloc_topology_get_complete_nodeset(top));
	hwloc_nodeset_t ns = hwloc_bitmap_alloc();
	hwloc_membind_policy_t* p = new hwloc_membind_policy_t;
	char* string;
	cout << "Getting Nodeset...";
	hwloc_get_area_membind_nodeset(top, _data, static_cast<size_t>(_size), ns, p, 0);
	cout << "Done"<<endl;
	int first = hwloc_bitmap_first(ns);
	cout << "First index is: " << first;
	hwloc_bitmap_asprintf(&string, ns);
	cout << string << endl;
	hwloc_bitmap_clr_range(ns, 0, -1);
	hwloc_bitmap_asprintf(&string, ns);
	cout << string << endl;

	hwloc_bitmap_set(ns, 5);
	hwloc_bitmap_asprintf(&string, ns);
	cout << string << endl;

	int ret = hwloc_set_area_membind_nodeset(top, _data, static_cast<size_t>(_size), ns, *p, HWLOC_MEMBIND_MIGRATE);
	cout << "Membind resulted in: " << ret << endl;
	hwloc_get_area_membind_nodeset(top, _data, static_cast<size_t>(_size), ns, p, 0);
	hwloc_bitmap_asprintf(&string, ns);
	cout << string << endl;

	delete p;
}

void hwloc_distribute() {
}

void hwloc_nexttouch(){
}

int main(int argc, char* argv[]) {
	assert(!(argc < 2));

	hwloc_topology_init(&top);
	hwloc_topology_load(top);

	_size = atoi(argv[1]) * sizeof(float);
	_data = static_cast<float*>(hwloc_alloc(top, _size));
	for ( int i = 0; i < atoi(argv[1]); i++ )
		_data[0] = 1;
	
	cout << "Simple Migration: " << endl;
	auto t1 = system_clock::now();
	hwloc_migrate();
	auto t2 = system_clock::now();
	cout << endl << "Time: " << duration_cast<microseconds>(t2-t1).count()<< "us" << endl; 
	
	t1 = system_clock::now();
	hwloc_distribute();
	t2 = system_clock::now();
	cout << "Distribution: ";	
	cout << duration_cast<microseconds>(t2-t1).count()<< endl; 

	t1 = system_clock::now();
	hwloc_nexttouch();
	t2 = system_clock::now();
	cout << "Next touch Migraion: ";	
	cout << duration_cast<microseconds>(t2-t1).count()<< endl; 
	
	hwloc_free(top, static_cast<void*>(_data), _size);
	hwloc_topology_destroy(top);
}
