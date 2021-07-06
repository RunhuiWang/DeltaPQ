
#include "pq.h"
#include "create_tree.h"

#include <algorithm> 
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
#include <fcntl.h>  // for direct I/O
#include <errno.h>

extern int PQ_M;
extern int PQ_K;
extern int with_id;
extern string ext;  // vector type, fvecs or bvecs
extern int dim; // dimension default is 128
#define BLKNBS 4096
#define BYTELEN 8
#define BLKNBITS BLKNBS*BYTELEN
#define NUM_DIM 8  // M
//#define NUM_DIFF 3
#define NUM_DIFF 8
#define EPSILON 0.000001
#define EPS 0.000001
long long int global_diff_sum = 0;
struct DummyNodes {
    int m = 8;
    uint size = 0;
    vector<uint> ids;   // vector ids

    void push_back(uint node_id) {
        ids.push_back(node_id);
        size ++;
    }
    void init(int M) {
        m = M;
        ids.resize(0);
        size = 0;
    }
    void init_nodes(const uchar* codes, long long num_codes, int M) {
        m = M;
        ids.resize(num_codes);
        for (long long i = 0; i < num_codes; i ++) {
            ids[i] = i;
        }
        size = num_codes;
    }
    void clear() {
        ids.resize(0); size = 0;
    }
};
vector<PQ::Array> global_m_codewords;
int global_m_Ds;
vector<float> global_query;
float** m_sub_distances=NULL;
float*** batch_m_sub_distances=NULL;
float** mkk_tables;
const uchar* pqcodes;
int n_queries=0;

struct QNode;
// Query Node
struct QNode {
    uint vec_id;
    uint parent_pos;
    uint child_pos_start;
    uint child_num;
    uint sub_tree_size=1; // including itself
    float qdist = 0.0;
    float max_dist=0;
    float max_dist2p=0;     // maximum distance to its parent
    
    uchar diff_num;
    uchar depth;
    array<Diff, NUM_DIFF> diffs;

    QNode(uint id, uint pid) : vec_id(id), parent_pos(pid), diff_num(0) { }
    QNode() : diff_num(0) {} 

    // set up vector id and parent position
    void set_id_parent_pos(uint id, uint p_pos) {
        vec_id = id; 
        parent_pos = p_pos;
    }
};

uchar* comprsd_dfs_codes;

vector<bool> is_active;
// for building trees
//void read_tree_index_file_approx(const string &file_name, QNode* nodes);
bool create_approx_tree(const string &dataset_path, const uchar* codes, 
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, uint num_codes, int diff_argument, 
                        int PART_NUM, 
                        QNode* nodes, vector<uint> &root_num_array,
                        float** m_sub_distance,
                        float** dist_tables, int max_height_folds,
                        int method = 1);
void find_edges_by_diff_approx(const string &dataset_path,
                        const uchar* codes, vector<uchar>& dummycodes, 
                        vector<bool>& dummymarks, int M, int K,
                        uint num_codes,uint& global_num_dummy,int diff_argument,
                        vector<pair<uint, uint>>& edges, float** dist_tables, 
                        uint& root_id, int max_height_folds,
                        int method = 1);
void partition_linear_opt_approx_with_constraint(const uchar* codes, 
                        vector<uchar>& dummycodes, vector<bool>& dummymarks, 
                        int M, int K, uint num_codes, 
                        uint& global_num_dummy, int DIFF_BY, uint* parents, 
                        uint* rank, vector<pair<uint, uint>>& edges, 
                        DummyNodes& dummy_nodes, DummyNodes& next,
                        DummyNodes& finalists, uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT);
void partition_linear_opt_approx_with_constraint_WOH(const uchar* codes, 
                        vector<uchar>& dummycodes, vector<bool>& dummymarks, 
                        int M, int K, uint num_codes, 
                        uint& global_num_dummy, int DIFF_BY, uint* parents, 
                        uint* rank, vector<pair<uint, uint>>& edges, 
                        DummyNodes& dummy_nodes, DummyNodes& next,
                        DummyNodes& finalists, uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT);
void partition_linear_opt_approx_clique_with_constraint_on_size(
                        const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, int nparts, uint num_codes,
                        uint& global_num_dummy, int DIFF_BY, uint* parents,
                        uint* rank, vector<pair<uint, uint>>& edges,
                        DummyNodes& dummy_nodes, DummyNodes& next,uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT, 
                        int min_clique_size);
void edges_to_tree_index_approx_dfs_layout(const string &dataset_path, 
                        const uchar* codes, vector<uchar>& dummycodes,
                        vector<bool>& dummymarks, int M, int K,
                        uint num_codes, uint global_num_dummy,
                        vector<pair<uint,uint>>& edges,
                        QNode* nodes, uint& root_id, int method);
bool qnodes_to_compressed_codes(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument,
                        long long n_diffs,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables, int method);
bool qnodes_to_compressed_codes_opt(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument,
                        long long n_diffs,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables, int method);
bool qnodes_to_compressed_codes_opt_block_aware(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument,
                        long long n_diffs,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables);
bool row_store_qnodes_to_compressed_codes_opt(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument,
                        long long n_diffs,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables, int method);
// for checking clique information
void check_clique_info(const uchar* codes, int M, int K, uint num_codes,
                        int DIFF_BY, uint* parents, uint* rank,
                        vector<pair<uint, uint>>& edges, float**  dist_tables);
void dfs(QNode* nodes, uint node_id, vector<pair<int, float>>& results, 
                                    uint& res_id, float dist, int depth);
void dfs(QNode* nodes, uint node_pos, vector<pair<int, float>>& results, 
                                    uint& res_idx, float dist);

float cal_dist_from_query(long long id, int PQ_M, const uchar* vecs, 
                            float** m_sub_distances);

float cal_distance_by_tables(uint a, uint b, float** dist_tables, const uchar* vecs, uint m_Ks) {
    float sum = 0;
    for (int m = 0; m < PQ_M; m ++) {
        int c_a = (int) vecs[(long long)a*PQ_M+m];
        int c_b = (int) vecs[(long long)b*PQ_M+m];
        sum += dist_tables[m][c_a*m_Ks+c_b]; // m_Ks = PQ_K
    }
    return sum;
}

long long check_num_diffs(const uchar* codes, int M, int K, int num_codes, vector<pair<uint, uint>>& edges) {

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
    int info_n=0;
    for (long long i = 0; i < edges.size(); i ++) {
        long long id_a = edges[i].first;
        long long id_b = edges[i].second;
        if (i < info_n) {
            cout << "ida " << id_a << " idb " << id_b << endl;
        }
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
            if (from != to) {
                if (i < info_n) cout << "(" << from << ", " << to <<") ";
                n_diffs++;
            }
        }
        if (i < info_n) cout << n_diffs << endl;
    }
    cout << "TOTAL number of diffs is " << n_diffs << endl;
    cout << "PQ_M = " << PQ_M << " K " << K << endl;
    return n_diffs;
}

void read_qnodes_from_file(const string &dataset_path, const uchar* codes,
                        int M, int K, uint num_codes, int diff_argument,
                        QNode* nodes, int method) {
    // Get QNodes first
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_TreeNodesDFS";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        ifstream ifs(file_name, ios::binary);
        if (!ifs.is_open()) {
            cerr << "Error: cannot open QNodes from " << file_name << ends;
            assert(0);
        }
        // load tree from file
        ifs.read( (char*) &(nodes[0]), sizeof(QNode) * (num_codes+1));
        cout << "Nodes are read from file " << file_name << endl;
    }
}
/*
void partition_linear_opt_approx_with_constraint_bitset(const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, uint num_codes,
                        uint& global_num_dummy, int DIFF_BY, uint* parents,
                        uint* rank, vector<pair<uint, uint>>& edges,
                        vector<bool>& is_connected, uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT)
{
    int LOG_K = round(log2(K));

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

 
    cout << " DIFF_BY = " << DIFF_BY << endl;
    int nparts = M;
    assert(nparts >= DIFF_BY);
    nchoosek(nparts, nparts-DIFF_BY, combinations);

    cout << "For loop begins " << get_current_time_str() << endl;
    vector<pair<uint128_t, uint>> hash_array;
    cout << combinations.size() << " combination(s)" << endl;

    long long debug_clique_size_sum = 0;
    long long debug_diff_num_sum = 0;

    for (auto k = 0; k < combinations.size(); k++) {
        hash_array.resize(0);
        //cout << k << " th combination" << endl ;
        //if (num_codes >= 1000000000) cout << k << " " ;

        gettimeofday(&beg, NULL);

        //cout << "All node size " << dummy_nodes.size << endl;
        //cout << "global num dummy " << global_num_dummy << endl;
        for (long long l = 0; l < num_codes; l ++) {
            int sp_count = 0;
            uint128_t hash = 0x0000000000000000ULL;
            uint code_id = l;
            if (is_connected[code_id] == true) continue;

            for (auto it = combinations[k].begin(); it != combinations[k].end();
                                                                        it++) {
                //hash |= (static_cast<uint128_t>(codes[GetIndex(M, code_id, *it)]) << (LOG_K * (*it)));
                if (nparts <= M) {
                    long start = (*it)*(M/nparts);
                    long end = (*it+1)*(M/nparts);
                    for (auto iter = start; iter < end; iter ++) {
                        int cid;
                        if (K > 256) {
                            cid = ((uint16_t*)codes)[GetIndex(M, code_id, iter)];
                        } else {
                            cid = codes[GetIndex(M, code_id, iter)];
                        }
                        hash |= (static_cast<uint128_t>(codes[GetIndex(M, code_id, iter)]) << (LOG_K * (iter)));
                    }
                } else {
                    int num_part = nparts/M;
                    int m = (*it) / num_part;
                    int part = (*it) % num_part;
                    int val = (uchar)(codes[GetIndex(M, code_id, m)]);
                    int val_length = LOG_K / num_part;
                    val = (val >> (val_length*part)) % ((1<<val_length)-1);
                    //cout << (*it) << endl;
                    //cout << bitset<8>(val) << endl;
                    hash |= (static_cast<uint128_t>(val) << (val_length* (*it)));
                    //cout << bitset<64>((uint64_t)hash) << endl;
                }
            }
                    //exit(0);
            hash_array.emplace_back(hash, code_id);// put index of dummy_nodes here
                                        // as push_back() in DummyNode required
        }
        cout << k << " th combination with nodes number of " << hash_array.size() << endl ;
        //gettimeofday(&mid, NULL);
        
        //cout << "   calculate hash codes  " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        //cout << "hash array size " << hash_array.size() << endl;
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(),
                [](const pair<uint128_t, uint32_t>& a,
                    const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        //gettimeofday(&mid, NULL);
        //cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // traverse hash array
        for (uint i = 0; i < hash_array.size(); i ++) {
            uint end = i+1;
            for (; end < hash_array.size(); end++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            if (end == i+1) {
                continue;
            }
            debug_clique_size_sum += end-i;

            // process the clique
            // find the highest node as parent
            int max_height = -1;
            uint parent_id = 0;
            for (long long j = i; j < end; j ++) {
                uint code_id = hash_array[j].second;
                if ((int)(heights[code_id]) > max_height) {
                    max_height = heights[code_id];
                    parent_id = code_id;
                }
            }
            // find the second highest node
            int second_height = 0;
            for (long long j = i; j < end; j ++) {
                uint code_id = hash_array[j].second;
                if (code_id == parent_id) continue;
                if ((int)(heights[code_id]) > second_height) {
                    second_height = heights[code_id];
                }
            }
            if (second_height == max_height) heights[parent_id] ++;

            if (max_height >= MAX_HEIGHT-1) {
                is_connected[parent_id] = true;
            }
            root_id = parent_id;
            for (long long j = i; j < end; j ++) {
                uint code_id = hash_array[j].second;
                if (code_id == parent_id) continue;
                is_connected[code_id] = true;
                // create the edges
                edges.emplace_back(parent_id, code_id);
//                cout << edges.size() << ". <" << parent_id << ", " << code_id << ">: ";
                for (int m = 0; m < M; m ++) {
                    uint to = codes[GetIndex(M, code_id, m)];  
                    uint from = codes[GetIndex(M, parent_id, m)];
                    if (from != to) 
                    {
                        debug_diff_num_sum++;
//                        cout << "(" << from << ", " << to << ") ";
                    }
                }
//                cout << endl;
            }
            i = end - 1;
        }
        
        //cout << "sum of clique size is " << debug_clique_size_sum << endl;
        //gettimeofday(&mid, NULL);
        //cout << "   combination used " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //double vm, rss;
        //process_mem_usage(vm, rss);
        //cout << "   partition linear opt approx VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    }
    cout << "number of edges "<< edges.size() << endl;
    cout << "number of diffs "<< debug_diff_num_sum << endl;
    global_diff_sum += debug_diff_num_sum;
    cout << "GLOBAL number of diffs "<< global_diff_sum << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: "
        << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
        << "sec" <<endl;
    if (DIFF_BY == M) {
        cout << "   TOTAL number of Diffs " << global_diff_sum << endl;
    }
}
*/

void partition_linear_opt_approx_with_constraint(const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, uint num_codes,
                        uint& global_num_dummy, int DIFF_BY, uint* parents,
                        uint* rank, vector<pair<uint, uint>>& edges,
                        DummyNodes& dummy_nodes, DummyNodes& next, 
                        DummyNodes& finalists, uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT)
{
    int LOG_K = round(log2(K));

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

/** original method    
    assert(M >= DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);
*/
    cout << " DIFF_BY = " << DIFF_BY << endl;
    int nparts = M;
    assert(nparts >= DIFF_BY);
    nchoosek(nparts, nparts-DIFF_BY, combinations);

    cout << "For loop begins " << get_current_time_str() << endl;
    vector<pair<uint128_t, uint>> hash_array;
    //hash_array.resize(dummy_nodes.size);
    cout << combinations.size() << " combination(s)" << endl;

    long long debug_clique_size_sum = 0;
    long long debug_diff_num_sum = 0;
    // mark the process nodes
    vector<bool> is_merged(dummy_nodes.size, false);
    vector<uint> node_ids(dummy_nodes.size,0);
    for (auto k = 0; k < combinations.size(); k++) {
        //if (num_codes >= 1000000000) cout << k << " " ;

        gettimeofday(&beg, NULL);
        long long num_active_codes = 0;
        // get active code ids
        for (long long l = 0; l < dummy_nodes.size; l ++) {
            if (!is_merged[l]) {
                node_ids[num_active_codes++] = l;
            }
        }
        hash_array.resize(num_active_codes);
        cout << k << " th combination with nodes number of " << num_active_codes << endl ;
        #pragma omp parallel for
        for (long long l = 0; l < num_active_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            uint code_id = dummy_nodes.ids[node_ids[l]];

            for (auto it = combinations[k].begin(); it != combinations[k].end();
                                                                        it++) {
                //hash |= (static_cast<uint128_t>(codes[GetIndex(M, code_id, *it)]) << (LOG_K * (*it)));
                long start = (*it)*(M/nparts);
                long end = (*it+1)*(M/nparts);
                for (auto iter = start; iter < end; iter ++) {
                    int cid;
                    if (K > 256) {
                        cid = ((uint16_t*)codes)[GetIndex(M, code_id, iter)];
                    } else {
                        cid = codes[GetIndex(M, code_id, iter)];
                    }
                    hash |= (static_cast<uint128_t>(cid) << (LOG_K * (iter)));
                }
            }
                    //exit(0);
            hash_array[l] = make_pair(hash, node_ids[l]);// put index of dummy_nodes here
                                        // as push_back() in DummyNode required
        }
        //gettimeofday(&mid, NULL);
        
        //cout << "   calculate hash codes  " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        //cout << "hash array size " << hash_array.size() << endl;
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(),
                [](const pair<uint128_t, uint32_t>& a,
                    const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        //gettimeofday(&mid, NULL);
        //cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // traverse hash array
        for (uint i = 0; i < hash_array.size(); i ++) {
            uint end = i+1;
            for (; end < hash_array.size(); end++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            if (end == i+1) {
                continue;
            }
            debug_clique_size_sum += end-i;

            // process the clique
            // find the highest node as parent
            int max_height = -1;
            uint parent_node_id = 0;
            for (long long j = i; j < end; j ++) {
                uint code_id = dummy_nodes.ids[hash_array[j].second];
                if (!is_active[code_id]) {
                    cout << "Wrong!!" << endl;
                }
                if ((int)(heights[code_id]) > max_height) {
                    max_height = heights[code_id];
                    parent_node_id = hash_array[j].second;
                }
            }
            uint parent_code_id = dummy_nodes.ids[parent_node_id];
            // find the second highest node
            int second_height = 0;
            for (long long j = i; j < end; j ++) {
                uint code_id = dummy_nodes.ids[hash_array[j].second];
                if (code_id == parent_code_id) continue;
                if ((int)(heights[code_id]) > second_height) {
                    second_height = heights[code_id];
                }
            }
            if (second_height == max_height) heights[parent_code_id] ++;
            max_height ++;

            if (max_height >= MAX_HEIGHT-2) {
                finalists.push_back(parent_code_id);
                is_merged[parent_node_id] = true;
            }
            root_id = dummy_nodes.ids[parent_node_id];
            for (long long j = i; j < end; j ++) {
                uint node_id = hash_array[j].second;
                uint code_id = dummy_nodes.ids[node_id];
                if (node_id == parent_node_id) continue;
                is_merged[node_id] = true;
                is_active[code_id] = false;
                // create the edges
                if (!is_active[parent_code_id]) {
                    cout << "Wrong " << endl;
                }
                edges.emplace_back(parent_code_id, code_id);
//                cout << edges.size() << ". <" << parent_id << ", " << node_id << ">: ";
                for (int m = 0; m < M; m ++) {
                    uint to = codes[GetIndex(M, code_id, m)];  
                    uint from = codes[GetIndex(M, parent_code_id, m)];
                    if (from != to) 
                    {
                        debug_diff_num_sum++;
                    }
                }
//                cout << endl;
            }
            i = end - 1;
        }
        

        //cout << "sum of clique size is " << debug_clique_size_sum << endl;
        //gettimeofday(&mid, NULL);
        //cout << "   combination used " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //double vm, rss;
        //process_mem_usage(vm, rss);
        //cout << "   partition linear opt approx VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    }
    // put nodes in "next"
    for (uint i = 0; i < dummy_nodes.size; i ++) {
        if (is_merged[i]) continue;
        uint code_id = dummy_nodes.ids[i];
        next.push_back(code_id);
    }
    cout << "number of edges "<< edges.size() << endl;
    cout << "number of diffs "<< debug_diff_num_sum << endl;
    global_diff_sum += debug_diff_num_sum;
    cout << "GLOBAL number of diffs "<< global_diff_sum << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: "
        << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
        << "sec" <<endl;
}

void partition_linear_opt_approx_with_constraint_WOH(const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, uint num_codes,
                        uint& global_num_dummy, int DIFF_BY, uint* parents,
                        uint* rank, vector<pair<uint, uint>>& edges,
                        DummyNodes& dummy_nodes, DummyNodes& next,
                        DummyNodes& finalists, uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT)
{
    int LOG_K = round(log2(K));

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

/** original method    
    assert(M >= DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);
*/
    cout << " DIFF_BY = " << DIFF_BY << endl;
    int nparts = M;
    assert(nparts >= DIFF_BY);
    nchoosek(nparts, nparts-DIFF_BY, combinations);

    cout << "For loop begins " << get_current_time_str() << endl;
    vector<pair<uint128_t, uint>> hash_array;
    //hash_array.resize(dummy_nodes.size);
    cout << combinations.size() << " combination(s)" << endl;

    long long debug_clique_size_sum = 0;
    long long debug_diff_num_sum = 0;
    // mark the process nodes
    vector<bool> is_merged(dummy_nodes.size, false);
    vector<uint> node_ids(dummy_nodes.size,0);
    for (auto k = 0; k < combinations.size(); k++) {
        //if (num_codes >= 1000000000) cout << k << " " ;

        gettimeofday(&beg, NULL);
        long long num_active_codes = 0;
        // get active code ids
        for (long long l = 0; l < dummy_nodes.size; l ++) {
            if (!is_merged[l]) {
                node_ids[num_active_codes++] = l;
            }
        }
        hash_array.resize(num_active_codes);
        cout << k << " th combination with nodes number of " << num_active_codes << endl ;
        #pragma omp parallel for
        for (long long l = 0; l < num_active_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            uint code_id = dummy_nodes.ids[node_ids[l]];

            for (auto it = combinations[k].begin(); it != combinations[k].end();
                                                                        it++) {
                //hash |= (static_cast<uint128_t>(codes[GetIndex(M, code_id, *it)]) << (LOG_K * (*it)));
                long start = (*it)*(M/nparts);
                long end = (*it+1)*(M/nparts);
                for (auto iter = start; iter < end; iter ++) {
                    int cid;
                    if (K > 256) {
                        cid = ((uint16_t*)codes)[GetIndex(M, code_id, iter)];
                    } else {
                        cid = codes[GetIndex(M, code_id, iter)];
                    }
                    hash |= (static_cast<uint128_t>(cid) << (LOG_K * (iter)));
                }
            }
                    //exit(0);
            hash_array[l] = make_pair(hash, node_ids[l]);// put index of dummy_nodes here
                                        // as push_back() in DummyNode required
        }
        //gettimeofday(&mid, NULL);
        
        //cout << "   calculate hash codes  " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        //cout << "hash array size " << hash_array.size() << endl;
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(),
                [](const pair<uint128_t, uint32_t>& a,
                    const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        //gettimeofday(&mid, NULL);
        //cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // traverse hash array
        for (uint i = 0; i < hash_array.size(); i ++) {
            uint end = i+1;
            for (; end < hash_array.size(); end++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            if (end == i+1) {
                continue;
            }
            debug_clique_size_sum += end-i;

            // process the clique
            // Use the first node as the parent
            uint parent_node_id = hash_array[i].second;
            uint parent_code_id = dummy_nodes.ids[parent_node_id];
            for (long long j = i+1; j < end; j ++) {
                uint code_id = dummy_nodes.ids[hash_array[j].second];
                if ((int)(heights[code_id])+1 >  (int)(heights[parent_code_id])) {
                    heights[parent_code_id] = (int)(heights[code_id])+1;
                }
            }
            // 
            if ((int)(heights[parent_code_id]) >= MAX_HEIGHT - 2) {
                finalists.push_back(parent_code_id);
                is_merged[parent_node_id] = true;
            }

            root_id = dummy_nodes.ids[parent_node_id];
            for (long long j = i+1; j < end; j ++) {
                uint node_id = hash_array[j].second;
                uint code_id = dummy_nodes.ids[node_id];
                is_merged[node_id] = true;
                // create the edges
                edges.emplace_back(parent_code_id, code_id);
//                cout << edges.size() << ". <" << parent_id << ", " << node_id << ">: ";
                for (int m = 0; m < M; m ++) {
                    uint to = codes[GetIndex(M, code_id, m)];  
                    uint from = codes[GetIndex(M, parent_code_id, m)];
                    if (from != to) 
                    {
                        debug_diff_num_sum++;
                    }
                }
//                cout << endl;
            }
            i = end - 1;
        }
        //cout << "sum of clique size is " << debug_clique_size_sum << endl;
        //gettimeofday(&mid, NULL);
        //cout << "   combination used " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //double vm, rss;
        //process_mem_usage(vm, rss);
        //cout << "   partition linear opt approx VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    }
    // put nodes in "next"
    for (uint i = 0; i < dummy_nodes.size; i ++) {
        if (is_merged[i]) continue;
        uint code_id = dummy_nodes.ids[i];
        next.push_back(code_id);
    }
    cout << "number of edges "<< edges.size() << endl;
    cout << "number of diffs "<< debug_diff_num_sum << endl;
    global_diff_sum += debug_diff_num_sum;
    cout << "GLOBAL number of diffs "<< global_diff_sum << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: "
        << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
        << "sec" <<endl;
    if (DIFF_BY == M) {
        cout << "   TOTAL number of Diffs " << global_diff_sum << endl;
    }
}
/*
void partition_linear_opt_approx_clique_with_constraint_on_size(
                        const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, int nparts, uint num_codes,
                        uint& global_num_dummy, int DIFF_BY, uint* parents,
                        uint* rank, vector<pair<uint, uint>>& edges,
                        DummyNodes& dummy_nodes, DummyNodes& next,uint& root_id,
                        vector<uchar>& heights, int MAX_HEIGHT, 
                        int min_clique_size, long& last_vec_id)
{
    int LOG_K = round(log2(K));
    cout << "number of edges "<< edges.size() << endl;

    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    cout << "nparts = " << nparts << " DIFF_BY = " << DIFF_BY << endl;
    assert(nparts >= DIFF_BY);
    nchoosek(nparts, nparts-DIFF_BY, combinations);

    cout << "For loop begins " << get_current_time_str() << endl;
    vector<pair<uint128_t, uint>> hash_array;
    //hash_array.resize(dummy_nodes.size);
    cout << combinations.size() << " combination(s)" << endl;

    long long debug_clique_size_sum = 0;
    long long debug_diff_num_sum = 0;
    // mark the process nodes
    vector<bool> is_merged(num_codes*2);
    for (long i = 0; i < num_codes*2; i ++) is_merged[i] = false;
    long clique_idx = 0;
    vector<uint> clique_sizes(0);
    vector<long> clique_ids(num_codes, -1);

    for (auto k = 0; k < combinations.size(); k++) {
        hash_array.resize(0);

        gettimeofday(&beg, NULL);

        for (long long l = 0; l < dummy_nodes.size; l ++) {
            int sp_count = 0;
            uint128_t hash = 0x0000000000000000ULL;
            uint node_id = dummy_nodes.ids[l];
            if (is_merged[node_id] == true) continue;

            for (auto it = combinations[k].begin(); it != combinations[k].end();
                                                                        it++) {
                //hash |= (static_cast<uint128_t>(codes[GetIndex(M, node_id, *it)]) << (LOG_K * (*it)));
                if (nparts <= M) {
                    long start = (*it)*(M/nparts);
                    long end = (*it+1)*(M/nparts);
                    for (auto iter = start; iter < end; iter ++)
                        hash |= (static_cast<uint128_t>(codes[GetIndex(M, node_id, iter)]) << (LOG_K * (iter)));
                } else {
                    int num_part = nparts/M;
                    int m = (*it) / num_part;
                    int part = (*it) % num_part;
                    int val = (uchar)(codes[GetIndex(M, node_id, m)]);
                    int val_length = LOG_K / num_part;
                    val = (val >> (val_length*part)) % ((1<<val_length)-1);
                    //cout << (*it) << endl;
                    //cout << bitset<8>(val) << endl;
                    hash |= (static_cast<uint128_t>(val) << (val_length* (*it)));
                    //cout << bitset<64>((uint64_t)hash) << endl;
                }
            }
                    //exit(0);
            hash_array.emplace_back(hash, l);// put index of dummy_nodes here
                                        // as push_back() in DummyNode required
        }
        //gettimeofday(&mid, NULL);
        
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(),
                [](const pair<uint128_t, uint32_t>& a,
                    const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        //gettimeofday(&mid, NULL);
        //cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //gettimeofday(&beg, NULL);
        // traverse hash array
        for (uint i = 0; i < hash_array.size(); i ++) {
            uint end = i+1;
            for (; end < hash_array.size(); end++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            //if (end == i+1) {
            if (end <= i + min_clique_size) {
                continue;
            }
            debug_clique_size_sum += end - i;
            uint clique_size = end - i;
            uint cliq_id = clique_idx ++;
            clique_sizes.emplace_back(clique_size);
            for (uint iter = i; iter < end; iter ++) {
                uint vec_id = dummy_nodes.ids[hash_array[iter].second];
                if (clique_ids[vec_id] == -1) clique_ids[vec_id] = cliq_id;
                uint vcid = clique_ids[vec_id];
                if (clique_sizes[vcid] < clique_size)
                    clique_ids[vec_id] = cliq_id;
            }
            i = end - 1;
        }
        
//        cout << "sum of clique size is " << debug_clique_size_sum << endl;
        //gettimeofday(&mid, NULL);
        //cout << "   combination used " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        //double vm, rss;
        //process_mem_usage(vm, rss);
        //cout << "   partition linear opt approx VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    }
    // put nodes in "next"
    // generate candidate cliques and merge cliques
    vector<vector<uint>> cliques;
    cliques.resize(clique_idx);
    for (uint i = 0; i < dummy_nodes.size; i ++) {
        uint vec_id = dummy_nodes.ids[i];
        uint cliq_id = clique_ids[vec_id];
        if (cliq_id == (uint)0-1) continue;
        cliques[cliq_id].emplace_back(vec_id);
    }
    for (uint i = 0; i < cliques.size(); i ++) {
        if (cliques[i].size() >= min_clique_size) {
            // INFO
            if (cliques[i].size() < info_max_size) 
                info_clique_sizes[cliques[i].size()]++;
            // END OF INFO
            for (uint vec_id : cliques[i]) {
                if (last_vec_id == -1) {
                    // this is the first node in the first clique ever
                    last_vec_id = vec_id;
                    root_id = vec_id;
                } else {
                    // create an edge with last_vec_id for each vector
                    edges.emplace_back(last_vec_id, vec_id);
                    for (int m = 0; m < M; m ++) {
                        uint to = codes[GetIndex(M, last_vec_id, m)];  
                        uint from = codes[GetIndex(M, vec_id, m)];
                        if (from != to) 
                        {
                            debug_diff_num_sum++;
                        }
                    }
                }
                is_merged[vec_id] = true;
            }
            last_vec_id = cliques[i][cliques[i].size()-1];
        }
    }
    for (uint i = 0; i < dummy_nodes.size; i ++) {
        uint node_id = dummy_nodes.ids[i];
        if (is_merged[node_id]) continue;
        next.push_back(node_id);
    }
    cout << "number of edges "<< edges.size() << endl;
    cout << "number of diffs "<< debug_diff_num_sum << endl;
    global_diff_sum += debug_diff_num_sum;
    cout << "GLOBAL number of diffs "<< global_diff_sum << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: "
        << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
        << "sec" <<endl;
}
*/

// true: read from file
// false: newly created
bool create_approx_tree(const string &dataset_path, const uchar* codes,
                        vector<uchar>& dummycodes, vector<bool>& dummymarks,
                        int M, int K, uint num_codes, int diff_argument,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables, int max_height_folds,
                        int method = 1)
{
    cout << "create_tree_index_approx() in create_tree_approx.h called " << endl;
    m_sub_distances = m_sub_distances_;
    mkk_tables = dist_tables;
    pqcodes = codes;

    if (num_codes >= INT_MAX) {
        cout << "Number of codes is too large: " << num_codes << endl;
        cout << "Exit" <<  endl;
        exit(0);
    }
    if (codes == NULL) {
        cout << "Please generate compressed codes first" << endl;
        exit(0);
    }

    // transform codes into strips
    const uchar* transformed_codes = codes;
    uint num_dummies = 0;
//    create_part_tree_index_approx(dataset_path, transformed_codes, dummycodes,
//                                    dummymarks, M, K, num_codes, 
//                                    num_dummies, diff_argument,
//                                    nodes, dist_tables);
    {
        cout << "Build trees by diffs " << endl;

        timeval beg, mid, mid1, end; 
        gettimeofday(&beg, NULL);

        vector<pair<uint, uint>> edges;
        uint root_id;
        //find_edges_by_diff(codes, M, K, num_codes, NUM_DIFF, edges);
        find_edges_by_diff_approx(dataset_path, codes, dummycodes, dummymarks, 
                                    M, K, num_codes, num_dummies, 
                                    diff_argument, edges, dist_tables, 
                                    root_id, max_height_folds, method);

        cout << "found " << edges.size() << " edges" << endl;
        gettimeofday(&mid, NULL);
        
        cout << "   ++++find edges by diff in "
            <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;

        long long n_diffs = check_num_diffs(codes, M, K, num_codes, edges);

        cout << dataset_path << endl;
        
        if (nodes == NULL) {
            cout << "QNodes were NULL" << endl;
            nodes = new QNode[num_codes+1];
        }
        edges_to_tree_index_approx_dfs_layout(dataset_path, codes, dummycodes, 
                                    dummymarks, M, K, num_codes, 
                                    num_dummies, edges, nodes, root_id,
                                    method);
        cout << "--------------Qnodes is " << nodes << endl;
        cout << "Building trees done" << endl;
        double vm, rss;
        process_mem_usage(vm, rss);
        cout << "Build tree VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
        
        // convert qnodes to compressed codes
        // create compressed codes
        // scan the qnodes array and fill out an array
        // output number of codes as well
        cout << "--------------Qnodes is " << nodes << endl;
        //qnodes_to_compressed_codes(dataset_path, codes,
        
        if (method < 4) {
        // DTC method in paper
            qnodes_to_compressed_codes_opt(dataset_path, codes,
                        M, K, num_codes, diff_argument, n_diffs,
                        nodes, m_sub_distances_, dist_tables, method);
        } else {
            // method = 4, block based design
            qnodes_to_compressed_codes_opt_block_aware(dataset_path, codes,
                        M, K, num_codes, diff_argument, n_diffs,
                        nodes, m_sub_distances_, dist_tables);
        }
        // row store
        //row_store_qnodes_to_compressed_codes_opt(dataset_path, codes,
        //                M, K, num_codes, diff_argument, n_diffs,
        //                nodes, m_sub_distances_, dist_tables, method);
    }

    uint num_nodes = num_codes;

    //delete[] transformed_codes;
    return false;
}

void edges_to_adj_lists_approx(const int num_codes, vector<pair<uint, uint>> &edges, vector<uint> &sparse_row, vector<uint> &offsets)
{
    cout << "Total number of edges " << edges.size() << endl;
    
    // create adjacent lists
//    int n_edges = edges.size();
//    for (auto i = 0; i < n_edges; i++)
//        edges.emplace_back(edges[i].second, edges[i].first);

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

// use dfs to find max dist
void dfs_find_max_dist(const uchar* codes, vector<uint>& sparse_row, vector<uint>& offsets, 
                   uint source, uint vec_id, float& max_dist, int K) {
    uint start = offsets[vec_id], end = offsets[vec_id+1];
    for (uint it = start; it < end; it ++) {
        uint child_vid = sparse_row[it];
        float dist = cal_distance_by_tables(source, sparse_row[it], mkk_tables,
                                            codes, K);
        if (dist > max_dist) max_dist = dist;
        dfs_find_max_dist(codes, sparse_row, offsets, source, sparse_row[it], max_dist, K);
    }
}
//void dfs_wnode_layout_only(uint& node_id, uint parent_vid,
//                        vector<uint>& sparse_row, vector<uint>& offsets, 
//                        int M) {
//    uint start = offsets[parent_vid], end = offsets[parent_vid+1];
//    for (uint it = start; it < end; it ++) {
//        node_id = node_id + 1;
//        uint child_id = sparse_row[it];
//        for (int i = 0; i < M; i ++) {
//            wnodes[node_id].code[i] = pqcodes[GetIndex(M, child_id, i)];
//        }
//        dfs_wnode_layout_only(node_id, child_id, sparse_row, offsets,M);
//    }
//}
//void dfs_cnode_layout_only(uint& node_id, uint parent_vid,
//                        vector<uint>& sparse_row, vector<uint>& offsets, 
//                        uchar depth, int M) {
//    uint start = offsets[parent_vid], end = offsets[parent_vid+1];
//    depth = depth + 1;
//    for (uint it = start; it < end; it ++) {
//        node_id = node_id + 1;
//        uint child_id = sparse_row[it];
//        cnodes[node_id].depth = depth;
//        uchar num_diff = 0;
//        for (int m = 0; m < M; m ++) {
//            uint from = pqcodes[GetIndex(M, parent_vid, m)];
//            uint to   = pqcodes[GetIndex(M, child_id,  m)];
//            if (from != to) {
//                cnodes[node_id].diffs[num_diff].m    = m;
//                cnodes[node_id].diffs[num_diff].from = from;
//                cnodes[node_id].diffs[num_diff].to   = to;
//                num_diff ++;
//            }
//        }
//        cnodes[node_id].diff_num = num_diff;
//        dfs_cnode_layout_only(node_id, child_id, sparse_row, offsets,depth,M);
//    }
//}

void dfs_node_layout(QNode* nodes, uint& node_id, uint parent_vid,
                        vector<uint>& sparse_row, vector<uint>& offsets, 
                        uchar depth, int M) {
    uint parent_node_id = node_id;
    nodes[parent_node_id].child_pos_start = node_id + 1;
    uint start = offsets[parent_vid], end = offsets[parent_vid+1];
    depth = depth + 1;
    for (uint it = start; it < end; it ++) {
        node_id = node_id + 1;
        uint child_id = sparse_row[it];
        nodes[node_id].set_id_parent_pos(child_id, parent_node_id);
        nodes[node_id].depth = depth;
        uchar num_diff = 0;
        for (int m = 0; m < M; m ++) {
            uint from = pqcodes[GetIndex(M, parent_vid, m)];
            uint to   = pqcodes[GetIndex(M, child_id,  m)];
            if (from != to) {
                nodes[node_id].diffs[num_diff].m    = m;
                nodes[node_id].diffs[num_diff].from = from;
                nodes[node_id].diffs[num_diff].to   = to;
                num_diff ++;
            }
        }
        nodes[node_id].diff_num = num_diff;
        dfs_node_layout(nodes, node_id, child_id, sparse_row, offsets, depth, M);
    }
    nodes[parent_node_id].child_num = node_id - parent_node_id;
}

//void read_tree_index_file_approx(const string &file_name, 
//                        QNode* nodes)
//{
//    ifstream ifs(file_name, ios::binary);
//    if (!ifs.is_open()) {
//        cerr << "Error: cannot open " << file_name << ends;
//        assert(0);
//    }
//    
//    uint num_nodes = 0;
//    ifs.read(reinterpret_cast<char*> (&num_nodes), sizeof(uint));
//
//    ifs.read((char*)&(nodes[0]),sizeof(QNode)*(num_nodes+1));
//
//    ifs.close();
//
//    double vm, rss;
//    process_mem_usage(vm, rss);
//    cout << "read file VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
//}


void find_edges_by_diff_approx(const string &dataset_path, const uchar* codes, 
                        vector<uchar>& dummycodes,
                        vector<bool>& dummymarks, int M, int K,
                        uint num_codes, uint& global_num_dummy, 
                        int diff_argument, vector<pair<uint, uint>>& edges,
                        float** dist_tables, uint& root_id,
                        int max_height_folds, int method)
{
    string file_name = dataset_path + "/M" + to_string(PQ_M) + "K" + to_string(K)
                            + "H" + to_string(max_height_folds) 
                            + "_Approx_Edges";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        ifstream ifs(file_name, ios::binary);
        if (!ifs.is_open()) {
            cerr << "Error: cannot open " << file_name << ends;
            assert(0);
        }
        // load tree from file
        ifs.read(reinterpret_cast<char*> (&root_id), sizeof(uint));
        edges.resize(num_codes-1);
        ifs.read( (char*) &(edges[0]), sizeof(pair<uint, uint>) * edges.size());
        cout << "Edges are read from file " << file_name << endl;
        return ;
    }
    
    cout << "Find_edges start ------------ " << get_current_time_str() << endl;

    uint* parents=NULL;
    uint* rank=NULL;
    uint max_num_nodes = num_codes*2;
    uint globa_num_dummy = 0;
    // prepare dummy nodes
    DummyNodes nodes_0;
    nodes_0.init_nodes(codes, num_codes, M);
    DummyNodes nodes_1;
    nodes_1.init(M);

    DummyNodes& dummy_nodes = nodes_0;
    DummyNodes& next = nodes_1;
    DummyNodes finalists;
    vector<uchar> heights(num_codes, 0);
    vector<bool> is_connected(num_codes, 0);
    is_active = vector<bool>(num_codes, true);
    int MAX_HEIGHTS = M * max_height_folds;
    for (int diff = 0; diff <= diff_argument; diff ++) {
        //partition_linear_opt_approx_with_constraint_bitset(codes, dummycodes,
        //        dummymarks, M, K, num_codes, global_num_dummy,
        //        diff, parents, rank, edges, is_connected, root_id,
        //        heights, MAX_HEIGHTS);
        switch (method) {
            case 1:
                partition_linear_opt_approx_with_constraint(codes, dummycodes,
                        dummymarks, M, K, num_codes, global_num_dummy,
                        diff, parents, rank, edges, dummy_nodes, next,finalists,
                        root_id, heights, MAX_HEIGHTS);
                break;
            case 2:
                partition_linear_opt_approx_with_constraint_WOH(codes, dummycodes,
                        dummymarks, M, K, num_codes, global_num_dummy,
                        diff, parents, rank, edges, dummy_nodes, next,finalists,
                        root_id, heights, MAX_HEIGHTS);
                break;

        }
        DummyNodes& tmp = dummy_nodes;
        dummy_nodes = next;
        next = tmp;
        next.clear();
        // the termination condition
        if (dummy_nodes.size <= 1) break;
        cout << "dummy_nodes.size = " << dummy_nodes.size << endl;
    }
    cout << "root id after enumerating combinations " << root_id << endl;
    if (dummy_nodes.size > 0) {
        finalists.push_back(dummy_nodes.ids[0]);    // there will only be 1 node left
    }
    // handle the roots with height of MAX_HEIGHT - 2
    cout << "finalists size is " << finalists.size << " edges size " << edges.size() << endl;
    if (finalists.size > 0) {
        uint parent_code_id = finalists.ids[0];
        root_id = parent_code_id;
        for (long long i = 1; i < finalists.size; i ++) {
            uint code_id = finalists.ids[i];
            // create the edges
            edges.emplace_back(parent_code_id, code_id);
            for (int m = 0; m < M; m ++) {
                uint to = codes[GetIndex(M, code_id, m)];  
                uint from = codes[GetIndex(M, parent_code_id, m)];
                if (from != to) 
                {
                    global_diff_sum++;
                }
            }
        }
    }
    cout << "number of edges is " << edges.size() << endl;
    cout << "   ++++ TOTAL number of Diffs " << global_diff_sum << endl;
    
    delete[] parents;
    delete[] rank;
    // write edges to file
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open()) {
        cerr << "Error: cannot open " << file_name << ends;
        assert(0);
    }
    cout << file_name << " " << " opened" << endl;
    ofs.write(reinterpret_cast<char*> (&root_id), sizeof(uint));
    ofs.write((char*) &(edges[0]), sizeof(pair<uint, uint>) * edges.size());
    ofs.close();
    cout << file_name << " " << " written and closed" << endl;

    cout << "Find_edges end ------------- " << get_current_time_str() << endl;
}

void edges_to_tree_index_approx_dfs_layout(const string &dataset_path,
                        const uchar* codes, vector<uchar>& dummycodes,
                        vector<bool>& dummymarks, int M, int K,
                        uint num_codes, uint global_num_dummy,
                        vector<pair<uint,uint>>& edges,
                        QNode* nodes, uint& root_id, int method)
{
    cout << "Edges to tree index..." << endl;
    // check if nodes have been stored
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_TreeNodesDFS";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        ifstream ifs(file_name, ios::binary);
        if (!ifs.is_open()) {
            cerr << "Error: cannot open " << file_name << ends;
            assert(0);
        }
        // load tree from file
        ifs.read( (char*) &(nodes[0]), sizeof(QNode) * (num_codes+1));
        cout << "Nodes are read from file " << file_name << endl;
        return ;
    }
    cout << "Creating tree index..." << endl;
    vector<uint> sparse_row;
    vector<uint> offsets;
    
    long long n_diffs = 0;
    uint* parents = new uint[num_codes];
    for (uint i = 0; i < num_codes; i ++) parents[i] = (uint)0-1;
    for (auto edge : edges) {
        parents[edge.second] = edge.first;
    }
    edges_to_adj_lists_approx(num_codes, edges, sparse_row, offsets);
    edges.resize(0);

    // get the max distance to sub tree
    float* max_dists = new float[num_codes];
    memset(max_dists, 0, sizeof(float)*num_codes);
    float* max_dist2p = new float[num_codes];
    memset(max_dist2p, 0, sizeof(float)*num_codes);

    // calculate max distances
    if (m_sub_distances == NULL) {
        cout << "In function edges_to_tree_index_approx() : " << endl
             << "        m_sub_distances is NULL" << endl; 
        exit(0);
    }

    cout << " num_codes = " << num_codes << endl;
    cout << "root id is " << root_id << endl;
    for (uint vid = 0; vid < num_codes; vid ++) {
        //cout << vid << " " << endl;;
        uint parent = parents[vid];
        uint prev_parent = vid;
        int depth = 0;
        while (parent != (uint)0-1) {
            //cout << parent << " depth " << depth << "    ";
            if (depth++ >= 16) {
                cout << endl;
                break;
            }
            float dist = cal_distance_by_tables(vid, parent,
                                                mkk_tables, codes, K);
            if (dist > max_dists[parent]) {
                max_dists[parent] = dist;
            }
            if (dist > max_dist2p[prev_parent])
                max_dist2p[prev_parent] = dist;
            prev_parent = parent;
            parent = parents[parent];
        }
    }
    cout << "max dist to ancestors done" << endl;

    // sort the children for each node
    for (uint vid = 0; vid < num_codes; vid ++) {
        uint start = offsets[vid], end = offsets[vid+1];
        sort(sparse_row.begin()+start, sparse_row.begin()+end, [max_dist2p](const uint a, const uint b) {
            return max_dist2p[a] > max_dist2p[b];
        });
    }

    // build the tree again with new order
    // reset QNode array
    memset(nodes, 0, sizeof(QNode)*(num_codes+1));
    for (uint i = 0; i < num_codes + 1; i ++) {
        nodes[i].sub_tree_size = 1;
    }

    cout << "Root id is " << root_id << endl;
    // set up root diffs
    for (int m = 0; m < M; m ++) {
        nodes[0].diffs[m].m = m;
        nodes[0].diffs[m].from = -1;
        nodes[0].diffs[m].to = codes[GetIndex(M, root_id, m)];
    }
    nodes[0].vec_id = root_id;
    nodes[0].diff_num = M;
    nodes[0].parent_pos = -1;
    nodes[0].depth = 0;
    
    uint node_id = 0;
    dfs_node_layout(nodes, node_id, root_id, sparse_row, offsets, nodes[0].depth, M);

    cout << "Root info " << nodes[0].child_pos_start << " " 
         << nodes[0].child_num << endl;
    long long debug_n_diffs = 0;
    for (uint pos = 0; pos < num_codes; pos ++) {
        //nodes[pos].sub_tree_size ++;
        nodes[pos].max_dist = sqrt(max_dists[nodes[pos].vec_id]);
        nodes[pos].max_dist2p = sqrt(max_dist2p[nodes[pos].vec_id]);
        if (pos > 0) debug_n_diffs += nodes[pos].diff_num;
    }
    cout << "debug n diffs is " << debug_n_diffs << endl;

    // INFO get the histogram of depths distribution
    uint* info_depth_counts = new uint[M+2];
    memset(info_depth_counts, 0, sizeof(uint)*(M+2));
    for (uint i = 0; i < num_codes; i ++) {
        info_depth_counts[nodes[i].depth] ++;
    }
    for (int i = 0; i < M+2; i ++) {
        cout << info_depth_counts[i] << " nodes at depth " << i << endl;
    }

    cout << "The sub tree size of root is " << nodes[0].sub_tree_size << endl;
    cout << "The Max dist of root is " << nodes[0].max_dist<< endl;
    // calculate maximum path distances and child range
    cout << "TOTAL NUMBER OF DIFFS is " << n_diffs << endl;
    cout <<"Build trees end -----------"<<get_current_time_str()<< endl;
    //
    // write edges to file
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open()) {
        cerr << "Error: cannot open " << file_name << ends;
        assert(0);
    }
    cout << file_name << " " << " opened" << endl;
    ofs.write((char*) &(nodes[0]), sizeof(QNode) * (num_codes+1));
    ofs.close();
        cout << "---------------Qnodes is " << nodes << endl;
}
void set_bit(uchar* bytes, long long offset, int val) {
    long long byte_offset = offset / 8;
    int bit_offset = offset % 8;
    bytes[byte_offset] |= val << bit_offset;
}
inline int get_bit(uchar* bytes, long long offset) {
    long long byte_offset = offset / 8;
    int bit_offset = offset % 8;
    return (bytes[byte_offset] >> bit_offset) & 1;
}
bool qnodes_to_compressed_codes(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument, 
                        long long n_diffs,
                        QNode* nodes, float** m_sub_distances_,
                        float** dist_tables, int method)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        cout << "REMINDER: DPNode file exists. No need to rebuild!" << endl
            << "    " << file_name << endl;
        return true;
    }
    cout << "qnodes_to_compressed_codes() in create_tree_approx.h called " << endl;
    pqcodes = codes;

    if (num_codes >= INT_MAX) {
        cout << "Number of codes is too large: " << num_codes << endl;
        cout << "Exit" <<  endl;
        exit(0);
    }
    // Get QNodes first
    read_qnodes_from_file(dataset_path, codes, M, K, num_codes, 
                            diff_argument, nodes, method);
    long long n_bits = n_diffs * 8 + 11 * (long long)num_codes + 64;// 64 is the root code
    long long n_bytes = n_bits/8;
    if (n_bits % 8 != 0) n_bytes ++;
    uchar* compressed_codes = new uchar[n_bytes];
    memset(compressed_codes, 0, sizeof(uchar)*n_bytes);
    cout << "number of bits is " << n_bits << endl;
    // traverse the QNodes and set data in bits
    long long bit_offset = 0;
    int last_depth = -1;
    for (uint i = 0; i < num_codes; i ++) {
        QNode& node = nodes[i];
        int depth = node.depth;
        if (depth > last_depth) {
//            if (depth - last_depth != 1) {
//                cout << "Depth is wrong!!!!!!!!!!" << endl;
//            } else {
//            }
//                cout << "depth increase by " << depth - last_depth << endl;
        }
        last_depth = depth;
        // set depth first
        if (depth >= 8) {
            cout << "depth is " << depth << endl;
            exit(0);
        }
        for (int j = 0; j < 3; j ++) {
            set_bit(compressed_codes, bit_offset++, (depth>>j)&1);
        }
        // set bit map
        uchar bit_map = 0;
        for (int j = 0; j < node.diff_num; j ++) {
            bit_map = bit_map | (1 << node.diffs[j].m);
        }
        if (i < 3) cout << "ndiff " << (int)node.diff_num << " " << bitset<8>(bit_map) << endl;
        for (int j = 0; j < 8; j ++) {  // 8 = log(L)
            set_bit(compressed_codes, bit_offset++, (bit_map>>j)&1);
        }
        // set "to"
        for (int j = 0; j < node.diff_num; j ++) { // 8 = log(l)
            uchar cid = node.diffs[j].to;
            if (i < 3) {
                cout <<"    to: " << bitset<8>(cid) << " " << endl;;
            }
            for (int k = 0; k < 8; k ++) {
                set_bit(compressed_codes, bit_offset++, (cid>>k)&1);
            }
        }
    }
    cout << "bit offset is " << bit_offset << endl;
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open() ) {
        cerr<<"Error: cannot open" << file_name << ends;
        assert(0);
    }
    for (int i = 0; i < 20; i ++) {
        cout << bitset<8>(compressed_codes[i]) << endl;
    }
    cout << file_name << " " << " created" << endl;
    ofs.write((char*) &(n_bits), sizeof(long long));
    ofs.write((char*) &(compressed_codes[0]), sizeof(uchar) * (n_bytes));
    ofs.close();
    return false;
}

bool row_store_qnodes_to_compressed_codes_opt(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument, long long n_diffs,
                        QNode *nodes, float** m_sub_distances_,
                        float** dist_tables, int method)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    file_name = file_name + "_row_store";
    if (exists_test3(file_name)) {
        cout << "REMINDER: DPNode file exists. No need to rebuild!" << endl
            << "    " << file_name << endl;
        return true;
    }
    cout << "row_store_qnodes_to_compressed_codes() in create_tree_approx.h called " << endl;
    pqcodes = codes;

    if (num_codes >= INT_MAX) {
        cout << "Number of codes is too large: " << num_codes << endl;
        cout << "Exit" <<  endl;
        exit(0);
    }
    // Get QNodes first
    read_qnodes_from_file(dataset_path, codes, M, K, num_codes, 
                            diff_argument, nodes, method);
    long long n_bytes = 8 + n_diffs + (3*((long long)num_codes-1)+1)/2;// 8 is the root code
    // row store
    if (ext == "fvecs") {
        n_bytes += (long long)num_codes * 4 * dim;
    } else {    // bvecs
        n_bytes += (long long)num_codes * dim;
    }// end row store
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "Before allocating compressed codes array  VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    cout << "number of bytes is " << n_bytes << endl;
    uchar* compressed_codes = new uchar[n_bytes];
    memset(compressed_codes, 0, sizeof(uchar)*n_bytes);
    // traverse the QNodes and set data in bits
    long long byte_offset = 0;
    for (int m = 0; m < M; m ++) {
        compressed_codes[byte_offset++] = nodes[0].diffs[m].to;
    }
    uint i = 1;
    long long debug_n_diffs = 0;
    for ( ; i < num_codes - 1; i += 2) {
        QNode& node1 = nodes[i];
        QNode& node2 = nodes[i+1];
        debug_n_diffs += (node1.diff_num + node2.diff_num);
        // set depth first
        uchar depths = node1.depth;
        depths = depths | ((node2.depth) << 4);
        if (i < 10) {
            cout << "depth1 = " << bitset<8>(node1.depth) << " depth2 = " 
                    << bitset<8>(node2.depth)
                    << " depths = " << bitset<8>(depths) << endl;
        }
        compressed_codes[byte_offset++] = depths;
        // -------- write node1
        // set bit map
        uchar bit_map = 0;
        for (int j = 0; j < node1.diff_num; j ++) {
            bit_map = bit_map | (1 << node1.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node1.diff_num; j ++) { // 8 = log(l)
            uchar cid = node1.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
        // -------- write node2
        bit_map = 0;
        for (int j = 0; j < node2.diff_num; j ++) {
            bit_map = bit_map | (1 << node2.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node2.diff_num; j ++) {
            uchar cid = node2.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
        // row store 
        if (ext == "fvecs") {
            byte_offset += 4*dim*2;
        } else { // bvecs
            byte_offset += dim*2;
        } // end row store
    }
    if ( i == num_codes - 1) {
        cout << " one code left byte_offset is " << byte_offset << endl;
        // a depth takes 8 bit
        compressed_codes[byte_offset++] = nodes[i].depth;
        uchar bit_map = 0;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            bit_map = bit_map | (1 << nodes[i].diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            compressed_codes[byte_offset++] = nodes[i].diffs[j].to;
        }
        debug_n_diffs += nodes[i].diff_num;
    }
    cout << "debug_n_diffs " << debug_n_diffs << endl;
    cout << "byte offset is " << byte_offset << endl;
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open() ) {
        cerr<<"Error: cannot open" << file_name << ends;
        assert(0);
    }
    for (int i = 0; i < 20; i ++) {
        cout << bitset<8>(compressed_codes[i])
            << " " << (int)compressed_codes[i] << endl;
    }
    cout << file_name << " " << " created" << endl;
    long long n_codes = num_codes;
    ofs.write((char*) &(n_codes), sizeof(long long));
    ofs.write((char*) &(n_bytes), sizeof(long long));
    ofs.write((char*) &(compressed_codes[0]), sizeof(uchar) * (n_bytes));
    ofs.close();
    return false;
}
bool qnodes_to_compressed_codes_opt(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument, long long n_diffs,
                        QNode *nodes, float** m_sub_distances_,
                        float** dist_tables, int method)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    if (with_id) file_name = file_name + "_with_id";
    switch (method) {
        case 1:
            break;
        case 2:
            file_name = file_name + "_WOH";
            break;
        case 3:
            file_name = file_name + "_clique";
            break;
    }
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        cout << "REMINDER: DPNode file exists. No need to rebuild!" << endl
            << "    " << file_name << endl;
        return true;
    }
    cout << "qnodes_to_compressed_codes() in create_tree_approx.h called " << endl;
    pqcodes = codes;

    if (num_codes >= INT_MAX) {
        cout << "Number of codes is too large: " << num_codes << endl;
        cout << "Exit" <<  endl;
        exit(0);
    }
    // Get QNodes first
    read_qnodes_from_file(dataset_path, codes, M, K, num_codes, 
                            diff_argument, nodes, method);
    long long n_bytes = 8 + n_diffs + (3*((long long)num_codes-1)+1)/2;// 8 is the root code
    uchar* compressed_codes = new uchar[n_bytes];
    memset(compressed_codes, 0, sizeof(uchar)*n_bytes);
    cout << "number of bytes is " << n_bytes << endl;
    // traverse the QNodes and set data in bits
    long long byte_offset = 0;
    for (int m = 0; m < M; m ++) {
        compressed_codes[byte_offset++] = nodes[0].diffs[m].to;
    }
    uint i = 1;
    long long debug_n_diffs = 0;
    for ( ; i < num_codes - 1; i += 2) {
        QNode& node1 = nodes[i];
        QNode& node2 = nodes[i+1];
        debug_n_diffs += (node1.diff_num + node2.diff_num);
        // set depth first
        uchar depths = node1.depth;
        depths = depths | ((node2.depth) << 4);
        if (i < 10) {
            cout << "depth1 = " << bitset<8>(node1.depth) << " depth2 = " 
                    << bitset<8>(node2.depth)
                    << " depths = " << bitset<8>(depths) << endl;
        }
        compressed_codes[byte_offset++] = depths;
        // -------- write node1
        // set bit map
        uchar bit_map = 0;
        for (int j = 0; j < node1.diff_num; j ++) {
            bit_map = bit_map | (1 << node1.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node1.diff_num; j ++) { // 8 = log(l)
            uchar cid = node1.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
        // -------- write node2
        bit_map = 0;
        for (int j = 0; j < node2.diff_num; j ++) {
            bit_map = bit_map | (1 << node2.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node2.diff_num; j ++) {
            uchar cid = node2.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
    }
    if ( i == num_codes - 1) {
        cout << " one code left byte_offset is " << byte_offset << endl;
        // a depth takes 8 bit
        compressed_codes[byte_offset++] = nodes[i].depth;
        uchar bit_map = 0;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            bit_map = bit_map | (1 << nodes[i].diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            compressed_codes[byte_offset++] = nodes[i].diffs[j].to;
        }
        debug_n_diffs += nodes[i].diff_num;
    }
    cout << "debug_n_diffs " << debug_n_diffs << endl;
    cout << "byte offset is " << byte_offset << endl;
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open() ) {
        cerr<<"Error: cannot open" << file_name << ends;
        assert(0);
    }
    for (int i = 0; i < 20; i ++) {
        cout << bitset<8>(compressed_codes[i])
            << " " << (int)compressed_codes[i] << endl;
    }
    cout << file_name << " " << " created" << endl;
    long long n_codes = num_codes;
    ofs.write((char*) &(n_codes), sizeof(long long));
    ofs.write((char*) &(n_bytes), sizeof(long long));
    ofs.write((char*) &(compressed_codes[0]), sizeof(uchar) * (n_bytes));
    ofs.close();
    return false;
}
bool qnodes_to_compressed_codes_opt_block_aware(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes, int diff_argument, long long n_diffs,
                        QNode *nodes, float** m_sub_distances_,
                        float** dist_tables)
{
    cout << "qnodes_to_compressed_codes() in deltapq_create_tree_approx.h called " << endl;
    pqcodes = codes;

    if (num_codes >= INT_MAX) {
        cout << "Number of codes is too large: " << num_codes << endl;
        cout << "Exit" <<  endl;
        exit(0);
    }
    // Get QNodes first
    read_qnodes_from_file(dataset_path, codes, M, K, num_codes, 
                            diff_argument, nodes, 1);
    long long n_bytes = 8 + n_diffs + (3*((long long)num_codes-1)+1)/2;// 8 is the root code
    uchar* compressed_codes = new uchar[n_bytes];
    memset(compressed_codes, 0, sizeof(uchar)*n_bytes);
    cout << "number of bytes is " << n_bytes << endl;
    // traverse the QNodes and set data in bits
    long long byte_offset = 0;
    for (int m = 0; m < M; m ++) {
        compressed_codes[byte_offset++] = nodes[0].diffs[m].to;
    }
    uint i = 1;
    long long debug_n_diffs = 0;
    int block_size_bits = 4096*8;
    long int n_blocks_used = 0;
    int block_offset = 0;
    block_offset += 10; // 10 bits for number of nodes in this block, max value is 4096*8/10
    for ( ; i < num_codes - 1; i += 2) {
        QNode& node1 = nodes[i];
        QNode& node2 = nodes[i+1];
        debug_n_diffs += (node1.diff_num + node2.diff_num);
        // calculate block usage for node1
        int expected_bits = 0;
        expected_bits += 10; // bitmap and two bits
        expected_bits += node1.diff_num * 8;
        if (block_offset + expected_bits > block_size_bits) {
            block_offset = 10 + expected_bits;
            n_blocks_used ++;
        } else {
            block_offset += expected_bits;
        }
        // calculate block usage for node2
        expected_bits = 0;
        expected_bits += 10; // bitmap and two bits
        expected_bits += node2.diff_num * 8;
        if (block_offset + expected_bits > block_size_bits) {
            block_offset = 10 + expected_bits;
            n_blocks_used ++;
        } else {
            block_offset += expected_bits;
        }

        // set depth first
        uchar depths = node1.depth;
        depths = depths | ((node2.depth) << 4);
        if (i < 10) {
            cout << "depth1 = " << bitset<8>(node1.depth) << " depth2 = " 
                    << bitset<8>(node2.depth)
                    << " depths = " << bitset<8>(depths) << endl;
        }
        compressed_codes[byte_offset++] = depths;
        // -------- write node1
        // set bit map
        uchar bit_map = 0;
        for (int j = 0; j < node1.diff_num; j ++) {
            bit_map = bit_map | (1 << node1.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node1.diff_num; j ++) { // 8 = log(l)
            uchar cid = node1.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
        // -------- write node2
        bit_map = 0;
        for (int j = 0; j < node2.diff_num; j ++) {
            bit_map = bit_map | (1 << node2.diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        // set "to"
        for (int j = 0; j < node2.diff_num; j ++) {
            uchar cid = node2.diffs[j].to;
            compressed_codes[byte_offset++] = cid;
        }
    }
    cout << "number of blocks used = " << n_blocks_used << endl;
    exit(0);

    if ( i == num_codes - 1) {
        cout << " one code left byte_offset is " << byte_offset << endl;
        // a depth takes 8 bit
        compressed_codes[byte_offset++] = nodes[i].depth;
        uchar bit_map = 0;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            bit_map = bit_map | (1 << nodes[i].diffs[j].m);
        }
        compressed_codes[byte_offset++] = bit_map;
        for (int j = 0; j < nodes[i].diff_num; j ++) {
            compressed_codes[byte_offset++] = nodes[i].diffs[j].to;
        }
        debug_n_diffs += nodes[i].diff_num;
    }
    cout << "debug_n_diffs " << debug_n_diffs << endl;
    cout << "byte offset is " << byte_offset << endl;
//    ofstream ofs(file_name, ios::binary);
//    if (!ofs.is_open() ) {
//        cerr<<"Error: cannot open" << file_name << ends;
//        assert(0);
//    }
//    for (int i = 0; i < 20; i ++) {
//        cout << bitset<8>(compressed_codes[i])
//            << " " << (int)compressed_codes[i] << endl;
//    }
//    cout << file_name << " " << " created" << endl;
//    long long n_codes = num_codes;
//    ofs.write((char*) &(n_codes), sizeof(long long));
//    ofs.write((char*) &(n_bytes), sizeof(long long));
//    ofs.write((char*) &(compressed_codes[0]), sizeof(uchar) * (n_bytes));
//    ofs.close();
    return false;
}

bool create_diff_index(const string &dataset_path, const uchar* codes,
                        int M, int K, uint& num_codes)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_diff_index";
    file_name = file_name + "_N" + to_string(num_codes);
    if (exists_test3(file_name)) {
        cout << "REMINDER: diff_index file exists. No need to rebuild!" << endl
            << "    " << file_name << endl;
        return true;
    }
    cout << "create_diff_index() in create_tree_approx.h called " << endl;
    pqcodes = codes;
    long long n_diffs = 8;
    for (long long i = 1 ; i < num_codes; i ++) {
        for (int m = 0; m < M; m ++) {
            if (K <= 256) { 
                if (codes[i*M+m] != codes[(i-1)*M+m]) n_diffs ++;
            } else {
                if (((uint16_t*)codes)[i*M+m] != ((uint16_t*)codes)[(i-1)*M+m])
                    n_diffs ++;
            }
        }
    }
    cout << "number of diffs is " << n_diffs << endl;
    long long n_bytes = 8 + n_diffs + num_codes-1;// 8 is the root code
    if (K > 256) n_bytes += n_diffs;
    uchar* diff_index = new uchar[n_bytes];
    memset(diff_index, 0, sizeof(uchar)*n_bytes);
    cout << "number of bytes is " << n_bytes << endl;
    // traverse the QNodes and set data in bits
    long long byte_offset = 0;
    for (int m = 0; m < M; m ++) {
        diff_index[byte_offset++] = codes[m];
    }
    long long i = 1;
    long long debug_n_diffs = 0;
    for ( ; i < num_codes ;i ++) {
        // -------- write node1
        // set bit map
        uchar bit_map = 0;
        for (int m = 0; m < M; m ++) {
            if (K <= 256) { 
                if (codes[i*M+m] != codes[(i-1)*M+m])
                    bit_map = bit_map | (1 << m);
            } else {
                if (((uint16_t*)codes)[i*M+m] != ((uint16_t*)codes)[(i-1)*M+m])
                    bit_map = bit_map | (1 << m);
            }
        }
        diff_index[byte_offset++] = bit_map;
        // set "to"
        for (int m = 0; m < M; m ++) {
            if (K <=256) {
                if (codes[i*M+m] != codes[(i-1)*M+m])
                    diff_index[byte_offset++] = codes[i*M+m];
            } else {
                if (((uint16_t*)codes)[i*M+m] != ((uint16_t*)codes)[(i-1)*M+m]){
                    diff_index[byte_offset++] = codes[(i*M+m)*2];
                    diff_index[byte_offset++] = codes[(i*M+m)*2+1];
                }
            }
        }
    }
    cout << "byte offset is " << byte_offset << endl;
    ofstream ofs(file_name, ios::binary);
    if (!ofs.is_open() ) {
        cerr<<"Error: cannot open" << file_name << ends;
        assert(0);
    }
    for (int i = 0; i < 20; i ++) {
        cout << bitset<8>(diff_index[i])
            << " " << (int)diff_index[i] << endl;
    }
    cout << file_name << " " << " created" << endl;
    long long n_codes = num_codes;
    ofs.write((char*) &(n_codes), sizeof(long long));
    ofs.write((char*) &(n_bytes), sizeof(long long));
    ofs.write((char*) &(diff_index[0]), sizeof(uchar) * (n_bytes));
    ofs.close();
    return false;
}
auto cmp_max = [](pair<float, uint>& left, pair<float, uint>& right) { 
    return (left.first) < (right.first);
};
inline int get_depth_from_compressed_codes(uchar* buffer, 
                                long long& bit_offset, FILE* &fp_codes,
                                long long max_n_bytes) {
    int depth = 0;
    long long boffset = bit_offset % (BLKNBITS);
//    cout << "in depth boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    if (boffset + 3 > BLKNBITS) {
        // need to read a new block
//        cout << "in get_depth =========== A New Block needed ===========" << endl;
        int breakpoint = BLKNBITS - boffset;
        for (int i = 0; i < breakpoint; i ++) {
            int val = get_bit(buffer, boffset+i);
            depth |= (val << i);
        }
        fread(buffer, sizeof(char), 
                min((long long)BLKNBS, 
                max_n_bytes-(bit_offset+breakpoint)/BYTELEN), 
                fp_codes);
        for (int i = 0; i < 3 - breakpoint; i ++) {
            int val = get_bit(buffer, i);
            depth |= (val << (i+breakpoint));
        }
        bit_offset += 3;
        return depth;
    }
    for (int i = 0; i < 3; i ++) {
        int val = get_bit(buffer, boffset+i);
        depth |= (val << i);
    }
    bit_offset += 3;
    if (boffset + 3 == BLKNBITS) {
//        cout << "in get_depth =========== A New Block needed after read ===========" 
//            << "block id " << (bit_offset+3) / (BLKNBITS)<< endl;
        if (bit_offset < max_n_bytes*BYTELEN) {
            fread(buffer, sizeof(char), 
                    min((long long)BLKNBS, max_n_bytes-(bit_offset/BYTELEN)), 
                    fp_codes);
        }
    }
    return depth;
    
}
inline int get_bitmap_from_compressed_codes(uchar* buffer,
                                long long& bit_offset, FILE* &fp_codes,
                                long long max_n_bytes) {
    int bitmap= 0;
    long long boffset = bit_offset % (BLKNBITS);
//    cout << "in bitmap boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    if (boffset + BYTELEN > BLKNBITS) {
        // need to read a new block
//        cout << "in get_bitmap =========== A New Block needed ===========" 
//            << "block id " << (bit_offset+BYTELEN) / (BLKNBITS)<< endl;
        int breakpoint = BLKNBITS - boffset;
        for (int i = 0; i < breakpoint; i ++) {
            int val = get_bit(buffer, boffset+i);
            bitmap |= (val << i);
        }
        fread(buffer, sizeof(char), 
                min((long long)BLKNBS, 
                max_n_bytes-(bit_offset+breakpoint)/BYTELEN), 
                fp_codes);
        for (int i = 0; i < BYTELEN - breakpoint; i ++) {
            int val = get_bit(buffer, i);
            bitmap |= (val << (i+breakpoint));
        }
        bit_offset += BYTELEN;
        return bitmap;
    }
    for (int i = 0; i < BYTELEN; i ++) {
        int val = get_bit(buffer, boffset+i);
        bitmap |= (val << i);
    }
    bit_offset += BYTELEN;
    if (boffset + BYTELEN == BLKNBITS) {
//        cout << "int get_bitmap =========== A New Block needed after read ===========" 
//            << "block id " << (bit_offset+BYTELEN) / (BLKNBITS)<< endl;
        if (bit_offset < max_n_bytes*BYTELEN) {
            fread(buffer, sizeof(char), 
                    min((long long)BLKNBS, max_n_bytes-(bit_offset/BYTELEN)), 
                    fp_codes);
        }
    }
    return bitmap;

}
inline int get_cid_from_compressed_codes(uchar* buffer,
                                long long& bit_offset, FILE* &fp_codes,
                                long long max_n_bytes) {
    int cid = 0;
    long long boffset = bit_offset % (BLKNBITS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    if (boffset + BYTELEN > BLKNBITS) {
        //INFO
//        cout << "int get_cid =========== A New Block needed ===========" 
//            << "block id " << (bit_offset+BYTELEN) / (BLKNBITS)<< endl;
        // need to read a new block
        int breakpoint = BLKNBITS - boffset;
        for (int i = 0; i < breakpoint; i ++) {
            int val = get_bit(buffer, boffset+i);
            cid |= (val << i);
        }
        long long nbyte_read = min((long long)BLKNBS, 
                max_n_bytes-(bit_offset+breakpoint)/BYTELEN);
//        cout << nbyte_read <<  " read " << endl;
        fread(buffer, sizeof(char), nbyte_read, fp_codes);
        for (int i = 0; i < BYTELEN - breakpoint; i ++) {
            int val = get_bit(buffer, i);
            cid |= (val << (i+breakpoint));
        }
//        //INFO
//        for (int i = 0; i < 20; i ++) {
//            cout << bitset<8>(buffer[i]) << endl;
//        }
//        // END OF INFO
        bit_offset += BYTELEN;
        return cid;
    }
    for (int i = 0; i < BYTELEN; i ++) {
        int val = get_bit(buffer, boffset+i);
        cid |= (val << i);
    }
    bit_offset += BYTELEN;
    if (boffset + BYTELEN == BLKNBITS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (bit_offset+BYTELEN) / (BLKNBITS)<< endl;
        if (bit_offset < max_n_bytes*BYTELEN) {
            fread(buffer, sizeof(char), 
                    min((long long)BLKNBS, max_n_bytes-(bit_offset/BYTELEN)), 
                    fp_codes);
        }
    }
    return cid;
}
void query_processing_scan_compressed_codes(const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes";
    file_name = file_name + "_N" + to_string(num_codes);
    FILE* fp_codes;
    uchar* buffer = new uchar[BLKNBS];
    cout << file_name << endl;
    fp_codes = fopen(file_name.c_str(), "r");
    fread(buffer, sizeof(char), sizeof(long long), fp_codes);
    long long n_bits = ((long long*)buffer)[0];
//    cout << n_bits << endl;
    long long n_bytes = n_bits/8 + ((n_bits%8==0) ? 0 : 1);
//    cout << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    // get first page of data
    fread(buffer, sizeof(char), BLKNBS, fp_codes);
    long long bit_offset=0;
    double qdist = 0;
    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<double> dists_stack(M,0);
    uchar root_bitmap = 0;
    bit_offset += 11;   // the first cid starts from offset 11
    for (int m = 0; m < M; m ++) {
        uchar cid = get_cid_from_compressed_codes(buffer, bit_offset, 
                                        fp_codes, n_bytes);
        qdist += m_sub_distances[m][cid];
        vecs_stack[0][m] = cid;
    }
    dists_stack[0] = qdist;
    max_heap.push(make_pair(qdist, 0));
    
    long long debug_diff_count = 0;
    int last_depth = 0;
    for (long i = 1; i < num_codes; i ++) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get depth
        int depth = get_depth_from_compressed_codes(buffer, 
                                            bit_offset, fp_codes, n_bytes);
        if (last_depth < depth) {
            if (last_depth + 1 != depth) {
                cout << last_depth << " " << depth << endl;
                exit(0);
            }
        }
        last_depth = depth;
        double dist = dists_stack[depth-1];
        uchar bitmap = get_bitmap_from_compressed_codes(buffer,
                                            bit_offset, fp_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // copy vector from parent
        for (int m = 0; m < M; m ++)
            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " ";
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_cid_from_compressed_codes(buffer,
                                            bit_offset, fp_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
//    cout << "bit offset is " << bit_offset << endl;
}
inline int get_cid_from_pqcodes(uchar* buffer,
                                long long& byte_offset, FILE* &fp_codes,
                                long long max_n_bytes) {
    int cid = 0;
    long long boffset = byte_offset % (BLKNBS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    cid = buffer[boffset];
    byte_offset += 1;
    if (boffset + 1 == BLKNBS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (byte_offset) / (BLKNBS)<< endl;
        if (byte_offset < max_n_bytes) {
            fread(buffer, sizeof(char), 
                    min((long long)BLKNBS, max_n_bytes-byte_offset), 
                    fp_codes);
        }
    }
    return cid;
}
void query_processing_scan_pqcodes(const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results)
{
    string file_name = dataset_path+"/codes.bin.plain.M"+to_string(M)
                        +"K"+to_string(K);
    FILE* fp_codes;
    uchar* buffer = new uchar[BLKNBS];
//    cout << file_name << endl;
    fp_codes = fopen(file_name.c_str(), "r");
    fread(buffer, sizeof(char), sizeof(long long), fp_codes);
    num_codes = ((long long*)buffer)[0];
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    // get first page of data
    fread(buffer, sizeof(char), BLKNBS, fp_codes);
    long long n_bytes = ((long long)num_codes) * M;
    
    long long byte_offset = 0;
    for (long i = 0; i < num_codes; i ++) {
        // calculdate distance to query
        float dist = 0;
        for (int m = 0; m < M; m ++) {
            uchar cid = get_cid_from_pqcodes(buffer,
                                            byte_offset, fp_codes, n_bytes);
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
}

//=========================== efficient decoding ========================
// this works for the compression with two 4-bit-depths in one byte
inline int get_byte_from_compressed_codes(uchar* buffer,
                                long long& byte_offset, FILE* &fp_codes,
                                long long max_n_bytes) {
    int cid = 0;
    long long boffset = byte_offset % (BLKNBS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    cid = buffer[boffset];
    byte_offset += 1;
    if (boffset + 1 == BLKNBS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (byte_offset) / (BLKNBS)<< endl;
        if (byte_offset < max_n_bytes) {
            fread(buffer, sizeof(char), 
                    min((long long)BLKNBS, max_n_bytes-byte_offset), 
                    fp_codes);
        }
    }
    return cid;
}
void query_processing_scan_compressed_codes_opt(const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    FILE* fp_codes;
    uchar* buffer = new uchar[BLKNBS];
    cout << file_name << endl;
    fp_codes = fopen(file_name.c_str(), "r");
    fread(buffer, sizeof(char), sizeof(long long)*2, fp_codes);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
//    cout << "n_codes = " << n_codes << endl;
//    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    // get first page of data
    fread(buffer, sizeof(char), BLKNBS, fp_codes);
    long long byte_offset=0;
    double qdist = 0;
    uchar* stacks = new uchar[M*M];
    uchar** vecs_stack = new uchar*[M];
    for (int i = 0; i < M; i ++) {
        vecs_stack[i] = stacks+i*M;
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<double> dists_stack(M,0);

    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fp_codes, n_bytes);
        qdist += m_sub_distances[m][cid];
        vecs_stack[0][m] = cid;
    }
    dists_stack[0] = qdist;
    max_heap.push(make_pair(qdist, 0));
    
    long i = 1;
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes(buffer, 
                                            byte_offset, fp_codes, n_bytes);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        double dist = dists_stack[depth-1];

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fp_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fp_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        dist = dists_stack[depth-1];

        bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fp_codes, n_bytes);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fp_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fp_codes, n_bytes);
        #pragma simd
        for (int m = 0; m < M; m ++)
            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        double dist = dists_stack[depth-1];
        uchar bitmap = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fp_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fp_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}


// ========================= O_Direct =======================
inline int get_cid_from_pqcodes(uchar* buffer,
                                long long& byte_offset, int& fd_codes,
                                long long max_n_bytes) {
    int cid = 0;
    long long boffset = byte_offset % (BLKNBS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    cid = buffer[boffset];
    byte_offset += 1;
    if (boffset + 1 == BLKNBS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (byte_offset) / (BLKNBS)<< endl;
        if (byte_offset < max_n_bytes) {
//            lseek (fd_codes, BLKNBS, SEEK_CUR);
            read(fd_codes, buffer, 
                    BLKNBS);
        }
    }
    return cid;
}
void query_processing_scan_pqcodes_o_direct(const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results)
{
    string file_name = dataset_path+"/codes.bin.plain.M"+to_string(M)
                        +"K"+to_string(K);
//    uchar* buffer = new uchar[BLKNBS];
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    memset(buffer, 0, BLKNBS);
    int fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int bytes_read = read(fd_codes, buffer, BLKNBS);
    if (bytes_read < 0) {
        cout << "bytes read " << bytes_read << endl;
        cout << "errno is " << errno << " " << strerror(errno) << endl;
    }
    num_codes = ((long long*)buffer)[0];
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    long long n_bytes = ((long long)num_codes) * M;
    
    long long byte_offset = 8;
    for (long i = 0; i < num_codes; i ++) {
        // calculdate distance to query
        float dist = 0;
        for (int m = 0; m < M; m ++) {
            uchar cid = get_cid_from_pqcodes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
}
inline uchar get_byte(uchar* buffer,
                      long long& byte_offset, int& fd_codes,
                      long long max_n_bytes) {
    uchar byte = 0;
    long long boffset = byte_offset % (BLKNBS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    byte = buffer[boffset];
    byte_offset += 1;
    if (boffset + 1 == BLKNBS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (byte_offset) / (BLKNBS)<< endl;
        if (byte_offset < max_n_bytes) {
//            lseek (fd_codes, BLKNBS, SEEK_CUR);
            read(fd_codes, buffer, 
                    BLKNBS);
        }
    }
    return byte;
}
void query_processing_diff_scan_o_direct(const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_diff_index";
    file_name = file_name + "_N" + to_string(num_codes);
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    memset(buffer, 0, BLKNBS);
    int fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int bytes_read = read(fd_codes, buffer, BLKNBS);
    if (bytes_read < 0) {
        cout << "bytes read " << bytes_read << endl;
        cout << "errno is " << errno << " " << strerror(errno) << endl;
    }
    num_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    long long byte_offset = 16;
    uchar* vec = new uchar[M];
    double qdist = 0;
    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte(buffer, byte_offset, fd_codes, n_bytes);
        vec[m] = cid;
        qdist += m_sub_distances[m][cid];
    }
    max_heap.emplace(qdist, 0);
    for (long i = 1; i < num_codes; i ++) {
        // calculdate distance to query
        uchar bitmap = get_byte(buffer, byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte(buffer, byte_offset, fd_codes, n_bytes);
            uchar from = vec[m];
            qdist -= m_sub_distances[m][from];
            qdist += m_sub_distances[m][cid];
            vec[m] = cid;
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(qdist, i);
        } else if (qdist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(qdist, i);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
}

// this works for the compression with two 4-bit-depths in one byte
inline int get_byte_from_compressed_codes(uchar* buffer,
                                long long& byte_offset, int &fd_codes,
                                long long max_n_bytes) {
    int cid = 0;
    long long boffset = byte_offset % (BLKNBS);
//    cout << "in cid boffset is " << boffset << " bit_offset is " 
//        << bit_offset << endl;
    cid = buffer[boffset];
    byte_offset += 1;
    if (boffset + 1 == BLKNBS) {
//        cout << "int get_cid =========== A New Block needed after read ===========" 
//            << "block id " << (byte_offset) / (BLKNBS)<< endl;
        if (byte_offset < max_n_bytes) {
            int byte_read = read(fd_codes, buffer, 
                    BLKNBS);
            if (max_n_bytes - byte_offset < BLKNBS) {
                cout << "The last block has size " << byte_read << endl;
            }
        }
    }
    return cid;
}
void query_processing_scan_compressed_codes_opt_o_direct(
            const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    int fd_codes;
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    cout << file_name << endl;
    fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int byte_read = read(fd_codes, buffer, BLKNBS);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    long long byte_offset=16;
    n_bytes += byte_offset;
    double qdist = 0;
    uchar* stacks = new uchar[M*M];
    uchar** vecs_stack = new uchar*[M];
    for (int i = 0; i < M; i ++) {
        vecs_stack[i] = stacks+i*M;
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<double> dists_stack(M,0);

    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fd_codes, n_bytes);
        qdist += m_sub_distances[m][cid];
        vecs_stack[0][m] = cid;
    }
    dists_stack[0] = qdist;
    max_heap.push(make_pair(qdist, 0));
    
    long i = 1;
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes(buffer, 
                                            byte_offset, fd_codes, n_bytes);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        double dist = dists_stack[depth-1];

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        dist = dists_stack[depth-1];

        bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        #pragma simd
        for (int m = 0; m < M; m ++)
            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        double dist = dists_stack[depth-1];
        uchar bitmap = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}
struct GreaterByDist
{
  bool operator()(const pair<float,uint>& a, const pair<float,uint>& b) const
  {
    return a.first < b.first;
  }
};
void query_processing_batch_scan_compressed_codes_opt_o_direct(
            const string &dataset_path,
            const vector<vector<float>> &queries, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<vector<pair<int, float>>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    int fd_codes;
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    cout << file_name << endl;
    fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int byte_read = read(fd_codes, buffer, BLKNBS);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (batch_m_sub_distances == NULL) {
        batch_m_sub_distances = new float**[queries.size()]; // m_sub_distances defined in .h file
        for (int q = 0; q < queries.size(); q ++) {
            batch_m_sub_distances[q] = new float*[PQ_M];
            for (int i = 0; i < PQ_M; i++) {
                batch_m_sub_distances[q][i] = new float[PQ_K];
                memset(batch_m_sub_distances[q][i], 0, sizeof(float)*PQ_K);
            }
        }
    }
    for (int q = 0; q < queries.size(); q ++) {
        const vector<float>& query = queries[q];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j ++) {
                batch_m_sub_distances[q][i][j] = .0;
                for (int k = 0; k < m_Ds; k ++) {
                    batch_m_sub_distances[q][i][j] += pow(m_codewords[i][j][k] 
                                            - query[i*m_Ds+k], 2);
                }
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    vector<priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        GreaterByDist>> batch_max_heap(queries.size());

    long long byte_offset=16;
    n_bytes += byte_offset;
    double * batch_qdist = new double[queries.size()];
    memset(batch_qdist, 0, sizeof(double)*queries.size());
    uchar* batch_stacks = new uchar[queries.size()*M*M];
    uchar*** batch_vecs_stack = new uchar**[queries.size()*M];
    for (int q = 0; q < queries.size(); q ++) {
        batch_vecs_stack[q] = new uchar*[M];
        for (int i = 0; i < M; i ++) {
            batch_vecs_stack[q][i] = batch_stacks + q*M*M +i*M;
        }
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<vector<double>> batch_dists_stack(queries.size());
    for (int q = 0; q < queries.size(); q ++) {
        batch_dists_stack[q].resize(M);
    }
    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fd_codes, n_bytes);
        for (int q = 0; q < queries.size(); q ++) {
            batch_qdist[q] += batch_m_sub_distances[q][m][cid];
            batch_vecs_stack[q][0][m] = cid;
        }
    }
    for (int q = 0; q < queries.size(); q ++) {
        batch_dists_stack[q][0] = batch_qdist[q];
        batch_max_heap[q].push(make_pair(batch_qdist[q], 0));
    }
    
    long i = 1;
    vector<float> batch_dist(queries.size());
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes(buffer, 
                                            byte_offset, fd_codes, n_bytes);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        for (int q = 0; q < queries.size(); q ++) {
            ((long long*)batch_vecs_stack[q][depth])[0] = ((long long*)batch_vecs_stack[q][depth-1])[0];
            batch_dist[q] = batch_dists_stack[q][depth-1];
        }

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[q][depth][m] = cid;
                uchar from = batch_vecs_stack[q][depth-1][m];
                batch_dist[q] -= batch_m_sub_distances[q][m][from];
                batch_dist[q] += batch_m_sub_distances[q][m][cid];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[q][depth] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        for (int q = 0; q < queries.size(); q ++) {
            ((long long*)batch_vecs_stack[q][depth])[0] = ((long long*)batch_vecs_stack[q][depth-1])[0];
            batch_dist[q] = batch_dists_stack[q][depth-1];
        }

        bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[q][depth][m] = cid;
                uchar from = batch_vecs_stack[q][depth-1][m];
                batch_dist[q] -= batch_m_sub_distances[q][m][from];
                batch_dist[q] += batch_m_sub_distances[q][m][cid];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[q][depth] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        for (int q = 0; q < queries.size(); q ++) {
            ((long long*)batch_vecs_stack[q][depth])[0] = ((long long*)batch_vecs_stack[q][depth-1])[0];
            batch_dist[q] = batch_dists_stack[q][depth-1];
        }

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[q][depth][m] = cid;
                uchar from = batch_vecs_stack[q][depth-1][m];
                batch_dist[q] -= batch_m_sub_distances[q][m][from];
                batch_dist[q] += batch_m_sub_distances[q][m][cid];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[q][depth] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
    }

    for (int q = 0; q < queries.size(); q ++) {
        for (int i = top_k-1; i >= 0; i --) {
            const pair<float, uint>& top = batch_max_heap[q].top();
            results[q][i].first = top.second;
            results[q][i].second= top.first;
            batch_max_heap[q].pop();
        }
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}
void query_processing_opt_batch_scan_compressed_codes_opt_o_direct(
            const string &dataset_path,
            const vector<vector<float>> &queries, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<vector<pair<int, float>>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    int fd_codes;
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    cout << file_name << endl;
    fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int byte_read = read(fd_codes, buffer, BLKNBS);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table, batch mode
    int nq = queries.size();
    vector<vector<float>> m_sub_distances_batch(PQ_M);
    for (int m = 0; m < M; m ++) {
        m_sub_distances_batch[m].resize( K * nq );
        for (int k = 0; k < K; k ++) {
            for (int q = 0; q < nq; q ++) {
                m_sub_distances_batch[m][k*nq+q] = 0;
                for (int d = 0; d < m_Ds; d ++) {
                    m_sub_distances_batch[m][k*nq+q] += 
                            pow(m_codewords[m][k][d] - queries[q][m*m_Ds+d], 2);
                }
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    vector<priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        GreaterByDist>> batch_max_heap(queries.size());

    long long byte_offset=16;
    n_bytes += byte_offset;
    double * batch_qdist = new double[queries.size()];
    memset(batch_qdist, 0, sizeof(double)*queries.size());
    uchar* batch_stacks = new uchar[queries.size()*M*M];
    uchar*** batch_vecs_stack = new uchar**[queries.size()*M];
    for (int m = 0; m < M; m ++) {
        batch_vecs_stack[m] = new uchar*[M];
        for (int mm = 0; mm < M; mm ++) {
            batch_vecs_stack[m][mm] = batch_stacks + m*M*nq + mm*nq;
        }
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<vector<double>> batch_dists_stack(M);
    for (int m = 0; m < M; m ++) {
        batch_dists_stack[m].resize(nq);
    }
    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fd_codes, n_bytes);
        for (int q = 0; q < queries.size(); q ++) {
            batch_qdist[q] += m_sub_distances_batch[m][cid*nq+q];
            batch_vecs_stack[0][m][q] = cid;
        }
    }
    for (int q = 0; q < queries.size(); q ++) {
        batch_dists_stack[0][q] = batch_qdist[q];
        batch_max_heap[q].push(make_pair(batch_qdist[q], 0));
    }
    
    long i = 1;
    vector<float> batch_dist(queries.size());
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes(buffer, 
                                            byte_offset, fd_codes, n_bytes);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
        memcpy(batch_vecs_stack[depth][0], batch_vecs_stack[depth-1][0], sizeof(uchar)*nq*M);
        #pragma simd
        for (int q = 0; q < queries.size(); q ++) {
            batch_dist[q] = batch_dists_stack[depth-1][q];
        }

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            #pragma simd
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[depth][m][q] = cid;
                uchar from = batch_vecs_stack[depth-1][m][q];
                batch_dist[q] -= m_sub_distances_batch[m][from*nq+q];
                batch_dist[q] += m_sub_distances_batch[m][cid*nq+q];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[depth][q] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        memcpy(batch_vecs_stack[depth][0], batch_vecs_stack[depth-1][0], sizeof(uchar)*nq*M);
        #pragma simd
        for (int q = 0; q < queries.size(); q ++) {
            batch_dist[q] = batch_dists_stack[depth-1][q];
        }

        bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            #pragma simd
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[depth][m][q] = cid;
                uchar from = batch_vecs_stack[depth-1][m][q];
                batch_dist[q] -= m_sub_distances_batch[m][from*nq+q];
                batch_dist[q] += m_sub_distances_batch[m][cid*nq+q];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[depth][q] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        memcpy(batch_vecs_stack[depth][0], batch_vecs_stack[depth-1][0], sizeof(uchar)*nq*M);
        for (int q = 0; q < queries.size(); q ++) {
            batch_dist[q] = batch_dists_stack[depth-1][q];
        }

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            for (int q = 0; q < queries.size();q++) {
                batch_vecs_stack[depth][m][q] = cid;
                uchar from = batch_vecs_stack[depth-1][m][q];
                batch_dist[q] -= m_sub_distances_batch[m][from*nq+q];
                batch_dist[q] += m_sub_distances_batch[m][cid*nq+q];
            }
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
        for (int q = 0; q < queries.size(); q ++) {
            batch_dists_stack[depth][q] = batch_dist[q];
        }
//        cout << endl;
//        cout << dist << " bit_offset " << bit_offset << endl;
        
        for (int q = 0; q < queries.size(); q ++) {
            if (batch_max_heap[q].size() < top_k) {
                batch_max_heap[q].emplace(batch_dist[q], i);
            } else if (batch_dist[q] < batch_max_heap[q].top().first) {
                batch_max_heap[q].pop();
                batch_max_heap[q].emplace(batch_dist[q], i);
            }
        }
    }

    for (int q = 0; q < queries.size(); q ++) {
        for (int i = top_k-1; i >= 0; i --) {
            const pair<float, uint>& top = batch_max_heap[q].top();
            results[q][i].first = top.second;
            results[q][i].second= top.first;
            batch_max_heap[q].pop();
        }
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}
// 
inline void skip_bytes_from_compressed_codes(int &fd_codes,
                                long long n_bytes_to_skip) {
    
}
void row_store_query_processing_scan_compressed_codes_opt_o_direct(
            const string &dataset_path,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    file_name = file_name + "_row_store";
    int fd_codes;
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    cout << file_name << endl;
    fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int byte_read = read(fd_codes, buffer, BLKNBS);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    long long byte_offset=16;
    n_bytes += byte_offset;
    double qdist = 0;
    uchar* stacks = new uchar[M*M];
    uchar** vecs_stack = new uchar*[M];
    for (int i = 0; i < M; i ++) {
        vecs_stack[i] = stacks+i*M;
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<double> dists_stack(M,0);

    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fd_codes, n_bytes);
        qdist += m_sub_distances[m][cid];
        vecs_stack[0][m] = cid;
    }
    dists_stack[0] = qdist;
    max_heap.push(make_pair(qdist, 0));
    
    long i = 1;
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes(buffer, 
                                            byte_offset, fd_codes, n_bytes);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        double dist = dists_stack[depth-1];

        uchar bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        dist = dists_stack[depth-1];

        bitmap = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
        // skip two raw data points
        if (ext == "fvecs") {
            for (int j = 0; j < dim*4*2; j ++) {
                get_byte_from_compressed_codes(buffer, byte_offset,
                                            fd_codes, n_bytes);
            }
        } else {    // bvecs
            for (int j = 0; j < dim*2; j ++) {
                get_byte_from_compressed_codes(buffer, byte_offset, 
                                            fd_codes, n_bytes);
            }
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        #pragma simd
        for (int m = 0; m < M; m ++)
            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        double dist = dists_stack[depth-1];
        uchar bitmap = get_byte_from_compressed_codes(buffer, byte_offset,
                                                fd_codes, n_bytes);
        int n_diff = decoder[bitmap][0];
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes(buffer,
                                            byte_offset, fd_codes, n_bytes);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}
void update_query_processing_simulation(
            const string &dataset_path,
            int M, int K, 
            int m_Ds, uint num_codes, uint num_updates,
            const vector<PQ::Array> &m_codewords, 
            uchar** decoder)
{
    string file_name = dataset_path + "/M" + to_string(M) + "K" + to_string(K)
                            + "_Approx_compressed_codes_opt";
    file_name = file_name + "_N" + to_string(num_codes);
    int fd_codes;
    uchar* buffer = (uchar*)aligned_alloc(BLKNBS, BLKNBS);
    cout << file_name << endl;
    fd_codes = open(file_name.c_str(), O_DIRECT|O_RDONLY);
    if (fd_codes < 0) {
        cout << "cannot open file " << file_name << endl;
    }
    int byte_read = read(fd_codes, buffer, BLKNBS);
    long long n_codes = ((long long*)buffer)[0];
    long long n_bytes = ((long long*)buffer)[1];
    if (num_codes == -1) num_codes = n_codes;
    else if (num_codes != n_codes) {
        cout << "scan only part of the codes " << num_codes << " / "
            << n_codes << endl;
    }
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;

    long long byte_offset=16;
    n_bytes += byte_offset;
    uchar* stacks = new uchar[M*M];
    uchar** vecs_stack = new uchar*[M];
    for (int i = 0; i < M; i ++) {
        vecs_stack[i] = stacks+i*M;
    }
    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes(buffer, byte_offset, 
                                        fd_codes, n_bytes);
        vecs_stack[0][m] = cid;
    }

    // update
    long long total_inc_diffs = 0;
    double total_inc_size = 0;
    n_bytes -= num_codes/4;
    for (int i = 0; i < num_updates; i ++) {
        int inc_diffs = 0;
        vector<uchar> new_vec(8);
        uchar bitmap = 0;
        for (int m = 0; m < M; m ++) {
            uchar cid = rand() % K;
            if (cid != vecs_stack[0][m]) {
                bitmap = bitmap | (1<<m);
                inc_diffs ++;
            }
            new_vec[0] = cid;
        }
        ofstream ofs(dataset_path + "/tmp", ios::binary);
        ofs.write(reinterpret_cast<char*> (&bitmap), sizeof(uchar));
        for (int m = 0; m < M; m ++) {
            uchar cid = new_vec[0];
            if (cid != vecs_stack[0][m])
                ofs.write(reinterpret_cast<char*> (&cid), sizeof(uchar));
        }
        ofs.close();
 
        total_inc_diffs += inc_diffs;
        total_inc_size += 1; // bitmap
        total_inc_size += inc_diffs; // diffs
        total_inc_size += 0.25; // two marking bits
        if (i % (num_updates/100) == 0) {
            cout << i << " "  <<  total_inc_diffs << " " << total_inc_size / n_bytes << endl;

        }
    }
    
}

//===================== In Memory Query ===================
inline int get_byte_from_compressed_codes_in_memory(uchar* buffer,
                                long long& byte_offset) {
    return buffer[byte_offset++];
}

void query_processing_scan_compressed_codes_opt_in_memory(
            uchar* codes, long long n_bytes,
            const vector<float> &query, int top_k, int M, int K, 
            int m_Ds, uint num_codes,
            const vector<PQ::Array> &m_codewords, 
            vector<pair<int, float>> &results, uchar** decoder)
{
    long long n_codes = num_codes;
    cout << "n_codes = " << n_codes << endl;
    cout << "n_bytes = " << n_bytes << endl;
//    cout << BLKNBITS<< endl;
    // calculate distance lookup table
    if (m_sub_distances == NULL) {
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j ++) {
            m_sub_distances[i][j] = .0;
            for (int k = 0; k < m_Ds; k ++) {
                m_sub_distances[i][j] += pow(m_codewords[i][j][k] 
                                        - query[i*m_Ds+k], 2);
            }
        }
    }
    // < <lowerbound, upperbound>, <distance, node_position> >
    priority_queue<pair<float, uint>, 
                    vector<pair<float, uint>>,
                        decltype(cmp_max)> max_heap(cmp_max);

    long long byte_offset=0;
    n_bytes += byte_offset;
    double qdist = 0;
    uchar* stacks = new uchar[M*M];
    uchar** vecs_stack = new uchar*[M];
    for (int i = 0; i < M; i ++) {
        vecs_stack[i] = stacks+i*M;
    }
//    vector<vector<uchar>> vecs_stack(M, vector<uchar>(M, 0));
    vector<double> dists_stack(M,0);

    for (int m = 0; m < M; m ++) {
        uchar cid = get_byte_from_compressed_codes_in_memory(codes, byte_offset);
        qdist += m_sub_distances[m][cid];
        vecs_stack[0][m] = cid;
        cout << (int)cid << " ";
    }
    cout << endl;
    dists_stack[0] = qdist;
    max_heap.push(make_pair(qdist, 0));
    
    long i = 1;
    for (; i+1 < num_codes; i = i + 2) {
//        cout << "============== i = " << i << " ===============" << endl;
        // get two depths
        int depths = get_byte_from_compressed_codes_in_memory(codes, 
                                            byte_offset);
//        cout << bitset<8>(depths) << " " << depths << endl;
        // -------------- PROCESS the first code in this pair
        int depth = depths & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        double dist = dists_stack[depth-1];

        uchar bitmap = get_byte_from_compressed_codes_in_memory(codes,
                                            byte_offset);
        int n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes_in_memory(codes,
                                            byte_offset);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i);
        }
        // -------------- PROCESS the second code in this pair
        depth = (depths>>4) & 7;
        // copy vector from parent
//        #pragma simd
//        for (int m = 0; m < M; m ++)
//            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        ((long long*)vecs_stack[depth])[0] = ((long long*)vecs_stack[depth-1])[0];
        dist = dists_stack[depth-1];

        bitmap = get_byte_from_compressed_codes_in_memory(codes,
                                            byte_offset);
        n_diff = decoder[bitmap][0];
        // calculdate distance to query
//        cout << "DEPTH " << depth << " n_diff = " << n_diff << " " << endl;
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes_in_memory(codes,
                                            byte_offset);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
//            cout << "(" << (int)from << "," << (int)cid << ") ";
        }
//        cout << endl;
        dists_stack[depth] = dist;
//        cout << dist << " bit_offset " << bit_offset << endl;
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }
    if (i == num_codes - 1) {
        // process one more code
        int depth = get_byte_from_compressed_codes_in_memory(codes, byte_offset);
        #pragma simd
        for (int m = 0; m < M; m ++)
            vecs_stack[depth][m] = vecs_stack[depth-1][m];
        double dist = dists_stack[depth-1];
        uchar bitmap = get_byte_from_compressed_codes_in_memory(codes, byte_offset);
        int n_diff = decoder[bitmap][0];
        for (int j = 0; j < n_diff; j ++) {
            int m = decoder[bitmap][j+1];
            uchar cid = get_byte_from_compressed_codes_in_memory(codes,
                                            byte_offset);
            vecs_stack[depth][m] = cid;
            uchar from = vecs_stack[depth-1][m];
            dist -= m_sub_distances[m][from];
            dist += m_sub_distances[m][cid];
        }
        if (max_heap.size() < top_k) {
            max_heap.emplace(dist, i+1);
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.emplace(dist, i+1);
        }
    }

    for (int i = top_k-1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i].first = top.second;
        results[i].second= top.first;
        max_heap.pop();
    }
//    cout << "byte offset after scan is " << byte_offset << endl;
}
