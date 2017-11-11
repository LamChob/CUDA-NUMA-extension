#ifndef NVECTOR_H
#define NVECTOR_H

#include <map>
#include <vector>
#include <initializer_list>
#include <mutex>
#include <hwloc.h>
#include <cstdint>

namespace nvec {
template <class T>
class nvector {
    public:
        typedef typename std::vector<T>::iterator iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;
        typedef std::initializer_list<int> init;

        nvector<T>(int size);
        nvector<T>(int size, T set);
	    nvector<T>(int size, Distribution&);
	    nvector<T>(int size, T set, Distribution&);

        ~nvector<T>();

        nvector<T>( const nvector& ); // copy
        nvector<T>( nvector&& ); // move
        nvector<T>& operator=( const nvector& ); // copy
        nvector<T>& operator=( nvector&& ); // move

        T& operator[](int inx) { return _data[inx]; };
        T& at(int);

        int get_num_domains(){ return _node_set.size(); };
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
		ndist* _dist;
};
#include <nvector.cpp>
} // end namespace
#endif
