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
		std::cerr << link_file << std::endl;
		int father, child;
		long long t;
		int num_edges = 0;
		while (f_stream >> father >> child >> t)
		{
			InsertEdge(father, child, t);
			num_edges++;
		}

	}

	static void InsertEdge(int father, int child, long long t)
	{
		edge_list[child].push_back(std::make_pair(father, t)); 
	}

	static void GetFathers(int child, long long t, std::vector<int>& fathers)
	{
		fathers.clear();
		for (size_t i = 0; i < edge_list[child].size(); ++i)
			if (edge_list[child][i].second <= t)
				fathers.push_back(edge_list[child][i].first);
	}

	static std::vector< std::vector< std::pair<int, long long> > > edge_list;
};

std::vector< std::vector< std::pair<int, long long> > > UserNet::edge_list;

#endif