#ifndef NDIST_H
#define NDIST_H

#include <map>
#include <vector>

namespace nvec{

constexpr int PAGE_SIZE {4096};
typedef std::vector<int> node_list; // TODO do smth. better...

class ndist {
	virtual void create_mapping(node_list nl,int n, unsigned int type_size)=0;	
	protected:
	std::map<int,int> mapping; // domain -> page map		
};

class block : public ndist {
	public:
		block(node_list nl, int n, unsigned int type_size);
		void create_mapping(node_list nl, int n, unsigned int type_size);
		std::map<int,int>& get_mapping();
};

class block_overlap : public ndist {
	public: 
		block_overlap(node_list nl, int n, unsigned int type_size, unsigned int overlap_range);
		void create_mapping(node_list nl, int n, unsigned int type_size, unsigned int overlap_range);
		std::map<int,int>& get_mapping();
};

class interleaved : public ndist {
	public:
		interleaved(node_list nl, int n, unsigned int type_size);
		void create_mapping(node_list nl, int n, unsigned int type_size);
		std::map<int,int>& get_mapping();
};

} // end namespace
#endif
