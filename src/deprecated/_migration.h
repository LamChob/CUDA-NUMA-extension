/*
 * Author: Dennis Sebastian Rieber, <rieber@stud.uni-heidelberg.de>
 * Generate utility for NUMA Migration of data
 */

#include <stddef.h>
#include <hwloc.h>
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

int migrate(void** d, size_t size, node_t node);
int set_membind_to_node(node_t node, membind_pol *reset);
int set_membind_policy(membind_pol *reset);
int _get_num_nodes();
int _get_pos(void*);
void _pin_to_node(node_t);
int _migrate_ft(void** d, size_t size, node_t node);
int _migrate_pm(void** d, size_t size, node_t node);
