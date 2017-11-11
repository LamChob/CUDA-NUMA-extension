#include <pthread.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <utility>
#include <stdexcept>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <thread>
#include <numavec.h>

#include <hwloc.h>


template<class T>
numavec<T>::numavec( int sz ) : _size(sz) {
    if ( _size < 0 ) { throw std::length_error("Size needs to be a positive integer") ; // throw if negativ _size indicator }

    _data.reserve(_size);
    _pagesize = sysconf(_SC_PAGESIZE);

	hwloc_topology_init(&top);
	hwloc_topology_load(top);
	hwloc_obj_t cs;
	for ( int i = 0; i < hwloc_get_nbobjs_by_type(top, HWLOC_OBJ_NUMANODE); i++ ) {
	    cs = hwloc_get_obj_by_type(top, HWLOC_OBJ_NUMANODE, i);
		  _node_set[i] = hwloc_bitmap_alloc();
		  hwloc_bitmap_copy(_node_set[i], cs->nodeset);
	}
}

template<class T>
numavec<T>::numavec( int sz , std::map<int, int> & destination): numavec(sz) {
   migrate(destination);
}

template<class T>
numavec<T>::numavec( int sz , T set): numavec(sz) {
  for(int i = 0; i < sz; i++) {
    _data[i] = set;
  }
}

template<class T>
numavec<T>::numavec( int sz , T set, std::map<int, int> & destination): numavec(sz,set) {
   migrate(destination);
}

template<class T>
numavec<T>::numavec( const numavec& other ) : numavec( other._size ) {
    for ( int i = 0 ; i < _size; i++ ) {
        _data[i] = other._data[i];
    }
    _destination = other._destination;

    std::thread t1 = std::thread(&numavec::_migrate,  this);
    t1.join();
}

template<class T>
numavec<T>& numavec<T>::operator=( const numavec& other ) {
    if ( this == &other )
         return *this;

    _size = other._size;
    hwloc_topology_init(&top);
    hwloc_topology_dup(top, other.top);

    for ( int i = 0 ; i < _size; i++ ) {
        _data[i] = other._data[i];
    }

    _destination = other._destination;

    std::thread t1 = std::thread(&numavec::_migrate,  this);
    t1.join();

    return *this;
}

// Move Constructor
template<class T>
numavec<T>::numavec( numavec&& other ) : numavec( other._size ) {
    _size = other._size;

    _data.clear();
    _data.shrink_to_fit();
    _data.reserve(_size);

    hwloc_topology_init(&top);
    hwloc_topology_dup(top, other.top);

    _data.swap(other._data);
    _destination = other._destination;

    other._size = 0;
    other._destination.clear();
    hwloc_topology_destroy(other.top);
}

template<class T>
numavec<T>& numavec<T>::operator=( numavec&& other ) {
    if ( this == &other )
        return *this;

    hwloc_topology_init(&top);
    hwloc_topology_dup(top, other.top);

    _size = other._size;
    _data.clear();
    _data.shrink_to_fit();
    _data.reserve(_size);

    _data.swap(other._data);

    _destination = other._destination;

    other._size = 0;
    other._destination.clear();
    hwloc_topology_destroy(other.top);

    return *this;
}


template<class T>
numavec<T>::~numavec() {
	for(auto e : _node_set)
		hwloc_bitmap_free(e.second);
  hwloc_topology_destroy(top);
}

template<class T>
T& numavec<T>::at( int i ) {
    std::lock_guard<std::mutex> lock(_data_lock);
    return _data.at(i);
}

templat<class T>
void numavec<T>::resize(size_t size) {
	std::lock_guard<std::mutex> lock(_data_lock);
	if (size == _data.size() ) return;
	_size = size;
	if (size < _data.size() ) { // decreasing size, so we have to get rid of the data
    	_data.resize(size);	
	} else {
		_data.reserve(size); // increasing size, only reserve additional space, to prevent data movement		
	}

	this->_migrate();	
}

template<class T>
void numavec<T>::migrate( int domain ) {
	std::lock_guard<std::mutex> lock(_data_lock);

	_destination.clear();
	_destination[domain] = 0;

	_migrate();
}

// TODO rewrite for new ordering
template<class T>
void numavec<T>::migrate( init list ) {
/*
    std::lock_guard<std::mutex> lock(_data_lock);

   int stepping = _size / _pagesize / list.size();
   stepping *= _pagesize;
   int lastpage = 0;
   for (auto e: list){
    	_destination[lastpage + stepping -1 ] = e;
	    lastpage += stepping;
   }
   //-- map last domain to last element
   int d = _destination[lastpage-1];
   _destination.erase(lastpage-1);
   _destination[_size] = d;

	std::thread t2 = std::thread(&numavec::_migrate, this);
	t2.join();
*/
}


template<class T>
void numavec<T>::_calculate_node_table( /*Usernode*/ ) {
	if ( _arrangement == EVEN ) {
		int elem_per_node = _size/usernodes.size();
		int inx = 0;
		for ( auto& e : usernodes ) {
			_destination[e] = inx;
			inx += elem_per_node;
		}
	} else if ( _arrangement == INTERLEAVED ) {
		// TODO add support later
	}
}


template<class T>
void numavec<T>::_migrate(){
	int start = 0;
    std::vector<std::thread> threads;

	for (auto& e: _destination){ 
        threads.push_back(std::thread(&numavec<T>::_migrate_part,this, e.first, start, e.second));
	    start = e.first + 1; 
	}

    for (auto& t: threads) {
        t.join();
    }
}


template<class T>
void numavec<T>::_core_pin( hwloc_nodeset_t cores ) {
 /*   cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET( core , &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);*/

	hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_cputset_from_nodeset(top, cpuset, cores); // convert correctly
	hwloc_set_cpubind(top, cpuset, HWLOC_CPUBIND_THREAD);
}

template<class T>
void numavec<T>::_migrate_part( int node, int start, int end) {
  hwloc_set_area_membind(top, &_data[start], (end-start)*sizeof(T), _node_set[node],
    HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET|HWLOC_MEMBIND_MIGRATE);
}
