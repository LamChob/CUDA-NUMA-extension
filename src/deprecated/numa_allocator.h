/*
 * numa_allocator.h
 * Author: Dennis Rieber (rieber@stud.uni-heidelberg.de)
 * Date: 28.11.2016
 *
 * Simple Allocator that uses hwlocs allocator functions. This makes later manipulations with hwloc 
 * easier
 *
 */
#include "hwloc.h"

#ifndef numa_allocator_h
#define numa_allocator_h

template <class T>
class numa_allocator {
	public:
		typedef T value_type;
		numa_allocator() {
			int err = hwloc_topology_init(&top);
			assert(!err);

			hwloc_topology_ignore_type(top, HWLOC_OBJ_PACKAGE);
			hwloc_topology_ignore_type(top, HWLOC_OBJ_CACHE);
			err = hwloc_topology_load(top);
			assert(!err);
		}; 

    	template<class O>
    	struct rebind { typedef numa_allocator<O> other; };

		numa_allocator(numa_allocator<T> const &na){
			hwloc_topology_dup(&top, na.top);
		};

		numa_allocator& operator=(numa_allocator<T> const&){};

		~numa_allocator() {
			hwloc_topology_destroy(top);
		};

		// check arg 2
		value_type* allocate(size_t n, const void* dist = 0) {
			return (value_type*) hwloc_alloc(top, n*sizeof(value_type));
		};
		
		void deallocate(void* p, size_t n) {
			hwloc_free(top, p, n*sizeof(value_type));
		}

	private: 
		hwloc_topology_t top;
};
#endif
