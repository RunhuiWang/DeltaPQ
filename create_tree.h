#ifndef CREATE_TREE_H
#define CREATE_TREE_H
//#include "pq.h"

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <inttypes.h>
#include <sys/time.h>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <parallel/algorithm>
#include <bitset>



#define NUM_DIM 8  // M
#define NUM_DIFF 8
string dataset_path;
//#define CHECK_CLIQUE 1

extern int with_id;
extern int PQ_M;
extern int PQ_K;

struct Diff{
    uchar m;
    uchar from;
    uchar to;
    Diff(){};
};

struct RootNode {
    uint vec_id;
    uint tree_size;
    array<uchar, NUM_DIM> code;
    RootNode(uint id) : vec_id(id), tree_size(0) { }
    RootNode() : tree_size(0) {} 
};

struct TreeNode {
    uint vec_id;
    uint parent_pos;
    // bool is_leaf;
    // bool is_lastchild;
    uchar diff_num;
    array<Diff, NUM_DIFF> diffs;
    TreeNode(uint id, uint pid) : vec_id(id), parent_pos(pid), diff_num(0) { }
    TreeNode() : diff_num(0) {} 
};

// for building trees
void read_tree_index_file(const string &file_name, const uint part_num, RootNode** roots_array, TreeNode** nodes_array, vector<uint> &root_num_array);
bool create_tree_index(const string &dataset_path, const uchar* codes, int M, int K, int num_codes, int diff_argument, int PART_NUM, RootNode **roots_array, TreeNode **nodes_array, vector<uint> &root_num_array, float** dist_table=NULL);
void create_part_tree_index(uchar* codes, int M, int K, int num_codes, int diff_argument, vector<RootNode> &roots, vector<TreeNode> &nodes, float** dist_table=NULL);
void find_edges_by_diff(uchar* codes, int M, int K, int num_codes, int diff_argument, vector<pair<uint, uint>>& edges, float** dist_tables);
void partition_linear_opt(uchar* codes, int M, int K, int num_codes, int DIFF_BY, uint* parents, uint* rank, vector<pair<uint, uint>>& edges);
void edges_to_tree_index(uchar* codes, int M, int num_cdoes, vector<pair<uint, uint>>& edges, vector<RootNode> &roots, vector<TreeNode> &nodes);
// for checking clique information
float cal_distance_by_tables(uint a, uint b, float** dist_tables, const uchar* vecs, int K);
void check_clique_info(uchar* codes, int M, int K, int num_codes, int DIFF_BY, uint* parents, uint* rank, vector<pair<uint, uint>>& edges, float**  dist_tables);

void check_num_diffs(uchar* codes, int M, int K, int num_codes, vector<pair<uint, uint>>& edges);

void nchoosek(int n, int k, vector<vector<int>> &combinations)
{
  vector<int> selected;
  vector<bool> selector(n, false);
  fill(selector.begin(), selector.begin() + k, true);
  do 
  {
    for (int i = 0; i < n; i++) 
    {
      if (selector[i])
        selected.push_back(i);
    }
    combinations.push_back(selected);
    selected.clear();
  } while (prev_permutation(selector.begin(), selector.end()));
}

inline uint64_t GetIndex(int M, uint64_t idx, int offset)
{
  return idx * M + offset;
}

// same usage as partition_linear_opt
void check_clique_info(uchar* codes, int M, int K, int num_codes, int DIFF_BY, uint* parents, uint* rank, vector<pair<uint, uint>>& edges, float** dist_tables)
{
    // stats arrays
    int n_bars = log10(num_codes);
    cout << n_bars << endl;
    long *size_histogram = new long[n_bars];
    memset(size_histogram, 0, sizeof(long long)*n_bars);
    for (int i = 0; i < n_bars; i ++) 
        cout << size_histogram[i] << " ";
    cout << endl;
    double avg_dist = 0;
    uint n_edges_added = 0;

    int LOG_K = round(log2(K));

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    vector<pair<uint128_t, uint>> hash_array;
    hash_array.resize(num_codes);
    cout << combinations.size() << " combinations" << endl;
    for (auto k = 0; k < combinations.size(); k++) {
        gettimeofday(&beg, NULL);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint128_t>(codes[GetIndex(M, l, *it)]) << (LOG_K * (*it)));
            hash_array[l] = make_pair(hash, l);
        }
        gettimeofday(&mid, NULL);
        
        gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(), [](const pair<uint128_t, uint32_t>& a, const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        gettimeofday(&mid, NULL);
        gettimeofday(&beg, NULL);
        // traverse hash array
        for (uint i = 0; i < num_codes; i ++) {
            uint end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            if (end == i+1) continue;
            size_histogram[(int)round(log10(end-i))]++;
            for (uint j = i; j < end-1; j ++) {
                uint start_i = hash_array[j].second;
                uint end_i = hash_array[j+1].second;
                uint x = find_set(parents, start_i);
                uint y = find_set(parents,end_i);
                //if (s_id >= e_id) continue;
                if (x != y) {
                    // add this edge into the tree
                    if (rank[x] > rank[y]) parents[y] = x;
                    else parents[x] = y;
                    if (rank[x] == rank[y]) rank[y] ++;

                    edges.emplace_back(end_i, start_i);
                    n_edges_added++;
                    avg_dist += cal_distance_by_tables(start_i, end_i, dist_tables, codes, K);
                }
            }
            i = end - 1;
        }

        gettimeofday(&mid, NULL);
    }
    cout << "-----------STATS size distribution ";
    for (int i = 0; i < n_bars; i ++) {
        cout << size_histogram[i] << " ";
    }
    cout << endl;
    cout << "-----------STATS avg. distance " << avg_dist / n_edges_added << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: " << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec - all_st.tv_usec) / 1e6 << "sec" <<endl;
}
void partition_linear_opt(uchar* codes, int M, int K, int num_codes, int DIFF_BY, uint* parents, uint* rank, vector<pair<uint, uint>>& edges)
{
    int LOG_K = round(log2(K));

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M >= DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    cout << "For loop begins " << get_current_time_str() << endl;
    vector<pair<uint128_t, uint>> hash_array;
    hash_array.resize(num_codes);
    cout << combinations.size() << " combinations" << endl;
    for (auto k = 0; k < combinations.size(); k++) {
        if (num_codes >= 1000000000) cout << k << " " ;

        gettimeofday(&beg, NULL);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++) {
                int cid;
                if (K > 256) {
                    cid = ((uint16_t*)codes)[GetIndex(M, l, *it)];
                } else {
                    cid = codes[GetIndex(M, l, *it)];
                }
                hash |= (static_cast<uint128_t>(cid) << (LOG_K * (*it)));
                //hash |= (static_cast<uint128_t>(codes[GetIndex(M, l, *it)]) << (LOG_K * (*it)));
            }
            hash_array[l] = make_pair(hash, l);
        }
        gettimeofday(&mid, NULL);
        
        //cout << "   calculate hash codes  " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(), [](const pair<uint128_t, uint32_t>& a, const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        gettimeofday(&mid, NULL);
        //cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        gettimeofday(&beg, NULL);
        // traverse hash array
        vector<vector<pair<uint, uint>>> candidate_edges(omp_get_max_threads());
        #pragma omp parallel for
        for (uint i = 1; i < num_codes; i++) 
        {
            if (hash_array[i].first == hash_array[i - 1].first)
            {
                if (find_set_read_only(parents, hash_array[i - 1].second)
                        != find_set_read_only(parents, hash_array[i].second))
                {
                    candidate_edges[omp_get_thread_num()].emplace_back(hash_array[i - 1].second, hash_array[i].second);
                }
            }
        }
        
        for (const auto &cand_edges : candidate_edges)
        {
            for (const auto &edge_pair : cand_edges)
            {
                uint x = find_set(parents, edge_pair.first);
                uint y = find_set(parents, edge_pair.second);
                if (x != y) {
                    // add this edge into the tree
                    if (rank[x] > rank[y]) parents[y] = x;
                    else parents[x] = y;
                    if (rank[x] == rank[y]) rank[y] ++;

                    edges.emplace_back(edge_pair);
                }
            }
        }

/*
        for (uint i = 0; i < num_codes; i ++) {
            uint end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            for (uint j = i; j < end-1; j ++) {
                uint start_i = hash_array[j].second;
                uint end_i = hash_array[j+1].second;
                uint x = find_set(parents, start_i);
                uint y = find_set(parents,end_i);
                //if (s_id >= e_id) continue;
                if (x != y) {
                    // add this edge into the tree
                    if (rank[x] > rank[y]) parents[y] = x;
                    else parents[x] = y;
                    if (rank[x] == rank[y]) rank[y] ++;

                    edges.emplace_back(end_i, start_i);
                }
            }
            i = end - 1;
        }
*/
        gettimeofday(&mid, NULL);
        //cout << "   find edges " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
    }
    cout << edges.size() << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: " << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec - all_st.tv_usec) / 1e6 << "sec" <<endl;
}


void query_processing(const vector<float> &query, int top_k, int M, int K, int m_Ds, int part_num, uint num_codes, RootNode **roots_array, TreeNode **nodes_array, const vector<uint> &root_num_array, vector<double> &dist_array, float** m_sub_distances, const vector<PQ::Array> &m_codewords, vector<pair<int, float>> &results)
{
    results.resize(top_k);

    int part_M = M / part_num;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] - query[i*m_Ds+k], 2);
            }
        }
    }

    for (int part_id = 0; part_id < part_num; part_id++)
    {
        int m_sub_dist_offset = part_id*part_M;
        uint root_num = root_num_array[part_id];
        const RootNode *roots = roots_array[part_id];
        const TreeNode *nodes = nodes_array[part_id];

        uint node_offset = 0;
        for (auto tree_id = 0; tree_id < root_num; tree_id++)
        {
            double rdist = 0;    // calculate query to root
            for (int m = part_id * part_M; m < (part_id + 1) * part_M; m++)
                rdist += m_sub_distances[m][roots[tree_id].code[m % part_M]];

            if (part_id == 0)
                dist_array[roots[tree_id].vec_id] = .0;

            dist_array[roots[tree_id].vec_id] += rdist;
            for (uint idx = 0; idx < roots[tree_id].tree_size-1; idx++)
            {
                uint node_id = idx + node_offset;
                float distance = dist_array[nodes[node_id].parent_pos];
                for (uchar diff_idx = 0; diff_idx < nodes[node_id].diff_num; diff_idx++) 
                {
                    distance -= m_sub_distances[nodes[node_id].diffs[diff_idx].m + m_sub_dist_offset][nodes[node_id].diffs[diff_idx].from];
                    distance += m_sub_distances[nodes[node_id].diffs[diff_idx].m + m_sub_dist_offset][nodes[node_id].diffs[diff_idx].to];
                }
                dist_array[nodes[node_id].vec_id] = distance;
            }
            node_offset += roots[tree_id].tree_size-1;
        }
    }

    double min_dist = FLT_MAX;
    int min_id = -1;
    for (uint vec_id = 0; vec_id < num_codes; vec_id++)
    {
        if (dist_array[vec_id] < min_dist)
        {
            min_dist = dist_array[vec_id];
            min_id = vec_id;
        }
    }

    results[0] = make_pair(min_id, min_dist);
}

// check stats of the MSTs
void check_msts(const vector<float> &query, int top_k, int M, int K, int m_Ds, int part_num, uint num_codes, RootNode **roots_array, TreeNode **nodes_array, const vector<uint> &root_num_array, vector<double> &dist_array, float** m_sub_distances, const vector<PQ::Array> &m_codewords, vector<pair<int, float>> &results)
{

    int part_M = M / part_num;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] - query[i*m_Ds+k], 2);
            }
        }
    }

    for (int part_id = 0; part_id < part_num; part_id++)
    {
        // get stats of this part tree
        int n_bars = log10(num_codes);
        int m_sub_dist_offset = part_id*part_M;
        uint root_num = root_num_array[part_id];
        const RootNode *roots = roots_array[part_id];
        const TreeNode *nodes = nodes_array[part_id];
        
        uint node_offset = 0;
        for (auto tree_id = 0; tree_id < root_num; tree_id++)
        {
            double rdist = 0;    // calculate query to root
            for (int m = part_id * part_M; m < (part_id + 1) * part_M; m++)
                rdist += m_sub_distances[m][roots[tree_id].code[m % part_M]];

            if (part_id == 0)
                dist_array[roots[tree_id].vec_id] = .0;

            dist_array[roots[tree_id].vec_id] += rdist;
            for (uint idx = 0; idx < roots[tree_id].tree_size-1; idx++)
            {
                uint node_id = idx + node_offset;
                float distance = dist_array[nodes[node_id].parent_pos];
                for (uchar diff_idx = 0; diff_idx < nodes[node_id].diff_num; diff_idx++) 
                {
                    distance -= m_sub_distances[nodes[node_id].diffs[diff_idx].m + m_sub_dist_offset][nodes[node_id].diffs[diff_idx].from];
                    distance += m_sub_distances[nodes[node_id].diffs[diff_idx].m + m_sub_dist_offset][nodes[node_id].diffs[diff_idx].to];
                }
                dist_array[nodes[node_id].vec_id] = distance;
            }
            node_offset += roots[tree_id].tree_size-1;
        }
    }

    double min_dist = FLT_MAX;
    int min_id = -1;
    for (uint vec_id = 0; vec_id < num_codes; vec_id++)
    {
        if (dist_array[vec_id] < min_dist)
        {
            min_dist = dist_array[vec_id];
            min_id = vec_id;
        }
    }

    results[0] = make_pair(min_id, min_dist);
}
// true: read from file
// false: newly created
bool create_tree_index(const string &_dataset_path, uchar* codes, int M, int K, int num_codes, int diff_argument, int part_num, RootNode **roots_array, TreeNode **nodes_array, vector<uint> &root_num_array, float** dist_tables=NULL)
{
    dataset_path = _dataset_path;   
    #ifndef CHECK_CLIQUE
    string file_name = _dataset_path + "/M" + to_string(PQ_M) + "K" + to_string(K) + "MultiMST";
//    if (diff_argument > 0) file_name = file_name + "_diff_" + to_string(diff_argument);
    if (with_id) file_name = file_name + "_with_id";
    cout << file_name << endl;
    if (exists_test3(file_name)) {
        // load tree from file
        read_tree_index_file(file_name, part_num, roots_array, nodes_array, root_num_array);
        return true;
    }
    #endif

    int part_M = M / part_num;
    assert(M % part_num == 0);
    long long array_length = (long long)num_codes * M;
    if (K > 256) array_length = array_length * 2;
    uchar* transformed_codes = new uchar[array_length];
    for (long long i = 0; i < num_codes; i ++) 
    {
        for (auto part_id = 0; part_id < part_num; part_id++) 
        {
            for (int m = part_id * part_M; m < (part_id + 1) * part_M; m++) 
            {
                if (K > 256) {
                    long long offsets = (long long)part_id * part_M * num_codes + i * part_M + m;
                    ((uint16_t*)transformed_codes)[offsets] = 
                                    ((uint16_t*)codes)[GetIndex(M, i, m)];
                } else {
                    transformed_codes[(long long)part_id * part_M * num_codes + i * part_M + m] = codes[GetIndex(M, i, m)];
                }
            }
        }
    }
    delete[] codes;

    for (int part_id = 0; part_id < part_num; part_id++)
    {
        vector<RootNode> roots;
        vector<TreeNode> nodes;

        create_part_tree_index(transformed_codes + (long long)part_id * num_codes * part_M, part_M, K, num_codes, diff_argument, roots, nodes, dist_tables);

        uint num_roots = roots.size();
        uint num_nodes = nodes.size();
        //roots_array[part_id] = new RootNode[num_roots];
        //nodes_array[part_id] = new TreeNode[num_nodes];
        ////cout << roots[0].vec_id << " " << roots[0].tree_size << " " << endl;
        ////cout << nodes[283].vec_id << " " << (int)(nodes[283].diff_num) << " " << endl;
        // The following line is used for query right after building
        //memcpy(roots_array[part_id], &(roots[0]), sizeof(RootNode)*num_roots);
        //memcpy(nodes_array[part_id], &(nodes[0]), sizeof(TreeNode)*num_nodes);
        //cout << roots_array[part_id][0].vec_id << " " << roots_array[part_id][0].tree_size << " " << endl;
        //cout << nodes_array[part_id][283].vec_id << " " << (int)(nodes_array[part_id][283].diff_num) << " " << endl;

//        #ifndef CHECK_CLIQUE
//        ofs.write(reinterpret_cast<char*> (&num_codes), sizeof(uint));
//        ofs.write(reinterpret_cast<char*> (&num_roots), sizeof(uint));
//        ofs.write(reinterpret_cast<char*> (&num_nodes), sizeof(uint));
//
//        root_num_array.push_back(num_roots);
//
//        ofs.write((char*) &(roots[0]), sizeof(RootNode) * num_roots);
//        ofs.write((char*) &(nodes[0]), sizeof(TreeNode) * num_nodes);
//        //ofs.write((char*) &(ptr_roots[0]), sizeof(RootNode) * num_roots);
//        //ofs.write((char*) &(ptr_nodes[0]), sizeof(TreeNode) * num_nodes);
//        #endif
//        cout << "--------------------------   Part " << part_id << " Processed" << 
//                "--------------------------" << endl;
    }
    /*
    for (int part_id = 0; part_id < part_num; part_id++)
    {
        vector<RootNode> roots;
        vector<TreeNode> nodes;

        create_part_tree_index(transformed_codes + part_id * num_codes * part_M, part_M, K, num_codes, diff_argument, roots, nodes);

        uint num_roots = roots.size();
        uint num_nodes = nodes.size();
        RootNode *ptr_roots = new RootNode[num_roots];
        TreeNode *ptr_nodes = new TreeNode[num_nodes];

        ofs.write(reinterpret_cast<char*> (&num_codes), sizeof(uint));
        ofs.write(reinterpret_cast<char*> (&num_roots), sizeof(uint));
        ofs.write(reinterpret_cast<char*> (&num_nodes), sizeof(uint));

        root_num_array.push_back(num_roots);

        ofs.write((char*) &(roots[0]), sizeof(RootNode) * num_roots);
        ofs.write((char*) &(nodes[0]), sizeof(TreeNode) * num_nodes);
        //ofs.write((char*) &(ptr_roots[0]), sizeof(RootNode) * num_roots);
        //ofs.write((char*) &(ptr_nodes[0]), sizeof(TreeNode) * num_nodes);

        delete[] ptr_roots;
        delete[] ptr_nodes;
    }
    */
    
    #ifndef CHECK_CLIQUE
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open()) {
        cerr << "Error: cannot open " << file_name << ends;
        assert(0);
    }
    cout << file_name << " " << " opened" << endl;
    #endif

    #ifndef CHECK_CLIQUE
    ofs.close();
    #endif

    delete[] transformed_codes;
    return false;
}

void create_part_tree_index(uchar* codes, int M, int K, int num_codes, int diff_argument, vector<RootNode> &roots, vector<TreeNode> &nodes, float** dist_tables)
{
    // prepare union-find set
    cout << "Build trees by diffs " << endl;

    timeval beg, mid, mid1, end; 
    gettimeofday(&beg, NULL);

    vector<pair<uint, uint>> edges;
    //find_edges_by_diff(codes, M, K, num_codes, NUM_DIFF, edges);
    find_edges_by_diff(codes, M, K, num_codes, diff_argument, edges, dist_tables);

    cout << "found " << edges.size() << " edges" << endl;
    gettimeofday(&mid, NULL);
    
    cout << "   ++++find edges by diff in " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;

    check_num_diffs(codes, M, K, num_codes, edges);
    
    edges_to_tree_index(codes, M, num_codes, edges, roots, nodes);

    cout << "Building trees done" << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "Build tree VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}

void edges_to_adj_lists(const int num_codes, vector<pair<uint, uint>> &edges, vector<uint> &sparse_row, vector<uint> &offsets)
{
    cout << "Total number of edges " << edges.size() << endl;
    
    // create adjacent lists
    int n_edges = edges.size();
    for (auto i = 0; i < n_edges; i++)
        edges.emplace_back(edges[i].second, edges[i].first);

    // sort edges
    sort(edges.begin(), edges.end(), [](const pair<uint, uint>& a, const pair<uint, uint>& b) {
        return a.first < b.first;
    });

    // sparse row contains all the "to"s
    // offset[x] contains num of edges from x;
    // edges are sorted first by from 

    sparse_row.clear(); // contains all tos
    offsets.clear(); // vector index is from_id, contains starting point

    vector<uint> num_neighbors(num_codes, 0);
    for (auto i = 0; i < edges.size(); i++)
    {
        uint parent = edges[i].first;
        uint child = edges[i].second;
        num_neighbors[parent]++;
        sparse_row.push_back(child);
    }

    uint idx = 0;
    offsets.push_back(idx);
    for (auto i = 0; i < num_codes; i++)
    {
        idx += num_neighbors[i];
        offsets.push_back(idx);
    }
}

void check_num_diffs(uchar* codes, int M, int K, int num_codes, vector<pair<uint, uint>>& edges) {

    for (long long i = 0; i < 10; i ++) {
        for (int m = 0; m < M; m ++) {
            if (K <= 256) {
                cout << (int)codes[(num_codes-1-i)*PQ_M+m] << " ";
            } else {
                //cout << (int)((uint16_t*)vecs)[(N-1-i)*PQ_M+m] << " ";
                cout << ((uint16_t*)codes)[GetIndex(PQ_M, num_codes - 1- i, m)] << " ";
            }
        }
        cout << endl;
    }
    // check total number of diffs
    long long n_diffs = 0;
    for (long long i = 0; i < edges.size(); i ++) {
        long long id_a = edges[i].first;
        long long id_b = edges[i].second;
        if (i < 10) cout << "ida " << id_a << " idb " << id_b << endl;
        for (int m = 0; m < M; m ++) {
            uint from;
            uint to;
            if (K > 256) {
                from = ((uint16_t*)codes)[GetIndex(M, id_a, m)];
                to   = ((uint16_t*)codes)[GetIndex(M, id_b, m)];
            } else {
                from = codes[GetIndex(M, id_a, m)];
                to   = codes[GetIndex(M, id_b, m)]; 
            }
            if (i < 10) cout << "(" << from << ", " << to <<") ";
            if (from != to)     n_diffs++;
        }
        if (i < 10) cout << "n_diffs = " << n_diffs << endl;
    }
    cout << "TOTAL number of diffs is " << n_diffs << endl;
    cout << "PQ_M = " << PQ_M << " K " << K << endl;

    exit(0);
}

// for NODE_PARENT_ID
void edges_to_tree_index(uchar* codes, int M, int num_codes, vector<pair<uint, uint>>& edges, vector<RootNode> &roots, vector<TreeNode> &nodes)
{
    vector<uint> sparse_row;
    vector<uint> offsets;
    edges_to_adj_lists(num_codes, edges, sparse_row, offsets);
    
    // to save memory
    edges.resize(0);

    vector<bool> tree_mark(num_codes, false); // because there are two directed edges for each pair
    queue<uint> bfs;
    long long n_diffs = 0;

    for (auto root_id = 0; root_id < num_codes; root_id++)
    {
        // a new tree
        if (tree_mark[root_id] == false)
        {
            //create a root node and add to bfs
            cout << "Root size is " << roots.size() << " rootid is " << root_id << endl;
            //roots.emplace_back(root_id);
            roots.emplace_back(root_id);

            for (auto m = 0; m < M; m++)
                roots.back().code[m] = codes[GetIndex(M, root_id, m)];

            bfs.push(root_id);
            tree_mark[root_id] = true;
            roots.back().tree_size++;

            while (bfs.empty() == false)
            {
                uint parent = bfs.front();
                bfs.pop();
                for (uint j = offsets[parent]; j < offsets[parent + 1]; j++)
                {
                    uint child = sparse_row[j];
                    if (tree_mark[child] == false)
                    {
                        nodes.emplace_back(child, parent); 
                        //nodes.push_back(TreeNode(child, parent));

                        uchar num_diff = 0;
                        for (int m = 0; m < M; m ++) 
                        {
                            uint from = codes[GetIndex(M, parent, m)];
                            uint to   = codes[GetIndex(M, child, m)]; 
                            if (from != to) 
                            {
                                nodes.back().diffs[num_diff].m    = m;
                                nodes.back().diffs[num_diff].from = from;
                                nodes.back().diffs[num_diff].to   = to;
                                num_diff++;

                            }
                        }
                        nodes.back().diff_num = num_diff;
                        bfs.push(child);
                        tree_mark[child] = true;
                        roots.back().tree_size++;

                        n_diffs += num_diff;
                    }
                }
            }
        }
    }

    long long max_size = 0;
    for (auto &r : roots)
        if (r.tree_size > max_size) max_size = r.tree_size;

    cout << "Largest MST tree has " << max_size << " nodes" << endl;
    cout << roots.size() << " trees are constructed " << endl;

    // calculate maximum path distances and child range
    cout << "TOTAL NUMBER OF DIFFS is " << n_diffs << endl;
    cout <<"Build trees end -----------"<<get_current_time_str()<< endl;
}


void read_tree_index_file(const string &file_name, const uint part_num, RootNode** roots_array, TreeNode** nodes_array, vector<uint> &root_num_array)
{
    ifstream ifs(file_name, ios::binary);
    if (!ifs.is_open()) {
        cerr << "Error: cannot open " << file_name << ends;
        assert(0);
    }
    
    for (int part_id = 0; part_id < part_num; part_id++) {

        uint num_codes = 0;
        uint num_roots = 0;
        uint num_nodes = 0;

        ifs.read(reinterpret_cast<char*> (&num_codes), sizeof(uint));
        ifs.read(reinterpret_cast<char*> (&num_roots), sizeof(uint));
        ifs.read(reinterpret_cast<char*> (&num_nodes), sizeof(uint));

        root_num_array.push_back(num_roots);
        roots_array[part_id] = new RootNode[num_roots];
        nodes_array[part_id] = new TreeNode[num_nodes];
    
        ifs.read( (char*) &(roots_array[part_id][0]), sizeof(RootNode) * num_roots);
        ifs.read( (char*) &(nodes_array[part_id][0]), sizeof(TreeNode) * num_nodes);
    }
    ifs.close();

    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "read file VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}

void find_edges_by_diff(uchar* codes, int M, int K, int num_codes, int diff_argument, vector<pair<uint, uint>>& edges, float** dist_tables)
{
    cout <<"Find_edges start -------------- "<<get_current_time_str()<< endl;
    string file_name = dataset_path + "/M" + to_string(PQ_M) + "K" + to_string(K)
                            +  "_MST_Edges";
    if (with_id) file_name = file_name + "_with_id";
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        ifstream ifs(file_name, ios::binary);
        if (!ifs.is_open()) {
            cerr << "Error: cannot open " << file_name << ends;
            assert(0);
        }
        // load tree from file
        edges.resize(num_codes-1);
        ifs.read( (char*) &(edges[0]), sizeof(pair<uint, uint>) * edges.size());
        cout << "Edges are read from file " << file_name << endl;
        return ;
    }
    cout << "Edges file not exists:  " << file_name << endl;

    uint* parents;
    uint* rank;
    parents = new uint[num_codes];
    rank = new uint[num_codes];
    for (uint i = 0; i < num_codes; i++) {
        parents[i] = i;
        rank[i] = 0;
    }
   
    for (int diff = 0; diff <= diff_argument; diff ++) {
        #ifdef CHECK_CLIQUE
        check_clique_info(codes, M, K, num_codes, diff, parents, rank, edges, dist_tables);
        #else
        partition_linear_opt(codes, M, K, num_codes, diff, parents, rank, edges);
        #endif
        if (edges.size() == num_codes - 1) {
            cout << "N-1 edges found in round diff " << diff << endl;
            break;
        }
    }
    delete[] parents;
    delete[] rank;
    // write edges to file
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open()) {
        cerr << "Error: cannot open " << file_name << ends;
        assert(0);
    }
    cout << file_name << " " << " opened" << endl;
    ofs.write((char*) &(edges[0]), sizeof(pair<uint, uint>) * edges.size());
    ofs.close();
    cout << file_name << " " << " written and closed" << endl;

    cout <<"Find_edges end ---------------"<<get_current_time_str()<< endl;
}
float cal_distance_by_tables(uint a, uint b, float** dist_tables, const uchar* vecs, int m_Ks) {
    float sum = 0;
    for (int m = 0; m < PQ_M; m ++) {
        int c_a = (int) vecs[(long long)a*PQ_M+m];
        int c_b = (int) vecs[(long long)b*PQ_M+m];
        sum += dist_tables[m][c_a*m_Ks+c_b]; // m_Ks = PQ_K
    }
    return sum;
}
#endif
