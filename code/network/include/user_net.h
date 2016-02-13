#ifndef USER_NET_H
#define USER_NET_H

#include <vector>
#include "config.h"
#include <fstream>

class UserNet
{
public:

	static void Init(unsigned max_user_cnt, const char* link_file) 
	{
		edge_list.clear();
		for (size_t i = 0; i < max_user_cnt; ++i)
		{
			edge_list.push_back(std::vector< std::pair<int, long long> >());
			edge_list[i].clear();
		}		
		std::ifstream f_stream(link_file);

		int father, child;
		long long t;
		while (f_stream >> father >> child >> t)
		{
			InsertEdge(father, child, t);
		}
	}

	static void InsertEdge(int father, int child, long long t)
	{
		edge_list[child].push_back(std::make_pair(father, t)); 
	}

	static std::vector< std::vector< std::pair<int, long long> > > edge_list;
};

std::vector< std::vector< std::pair<int, long long> > > UserNet::edge_list;

#endif