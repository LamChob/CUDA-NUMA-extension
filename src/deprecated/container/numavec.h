#ifndef NUMAVEC_H
#define NUMAVEC_H

#include <map>
#include <vector>
#include <initializer_list>
#include <mutex>
#include <hwloc.h>
#include <cstdint>

template <class T>
class numavec {
    public:

		// for simple migration patters
		enum class Distribution : std::int8 {
			EVEN = 0,
			INTERLEAVED = 1
		};

        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;
        typedef std::initializer_list<int> init;

        numavec<T>(int size);
        numavec<T>(int size, T set);
		    numavec<T>(int size, std::map<int, int>& distri);
		    numavec<T>(int size, T set,  std::map<int, int> &);

        ~numavec<T>();

        numavec<T>( const numavec& ); // copy
        numavec<T>( numavec&& ); // move
        numavec<T>& operator=( const numavec& ); // copy
        numavec<T>& operator=( numavec&& ); // move

        T& operator[](int in) { return _data[in]; };
        T& at(int);

        int get_num_domains(){ return _node_set.size(); };
        int get_elems_per_page(){ return _pagesize / sizeof(T); };
        int size() const { return _size; };
        const T* data() const { return _data.data(); };

        void migrate( int );
        void migrate( Arrangement );
        void migrate( init list );

		void resize(size_t size);

        const_iterator cbegin() const { return _data.cbegin(); };
        const_iterator cend() const { return _data.cbegin() + _size; };

        const_iterator begin() const {return _data.begin();};
        const_iterator end() const { return _data.begin() + _size; };

        iterator begin() {return _data.begin();};
        iterator end() { return _data.begin() + _size; };


    private:
        std::mutex _data_lock;
        std::vector<T> _data;
        int _pagesize;
        int _size;

        hwloc_topology_t top;

        std::map<int,hwloc_cpuset_t> _node_set;
        std::map<int,int>_destination; // <node:first index>
		Arrangement _arrangement;

        void _migrate( );
        void _core_pin(int);
        void _migrate_part(int node, int start, int end);
		void _calculate_node_table(); // calculate the which index belogs to which node

};
#include <numavec.cpp>

#endif
