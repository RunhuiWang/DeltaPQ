#ifndef PQ_BASE_H
#define PQ_BASE_H

#include <cassert>
#include <vector>
#include <set>
//#include <queue>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory.h>
#include <float.h>  // for FLT_MAX
/*the input edges (u,v) must be sorted by increasing order of u (tiers broken arbitarily if u is the same)!!!!
Otherwise, it will be incorrect*/
#include <queue>

#include "utils.h"
#include "find_edge.h"
#define NUM_DIFF 3

extern int PQ_M;
extern int PQ_K;
extern int rotate_tree_flag;

using namespace std;

typedef unsigned char uchar;
typedef vector<vector<float> > Array;

class PQBase {
public:
    //assume that # of nodes fits into long long, and # of edges fits into long long
    vector<vector<int>> g;
    vector<vector<int>> gr;
    vector<vector<pair<int, float>>> g_trees;
    vector<set<int>> g2;
    int d_in_max;
    string data_folder;

    // compressed sparse row
    vector<pair<uint, uint>> edges;
    uint* sparse_row = NULL;
    long* offsets = NULL;
    inline uint get_degree(uint i);
    inline uint get_neighbor(uint i, uint j);

    static bool cmp(const pair<int, double> &t1, const pair<int, double> &t2);

    long long n;
    long long m;

    long long N;    // The same as n
    int n_partition;    // used for finding edges

    vector<Array> m_codewords; // [ns][ks][ds]
    int m_M ; 
    int m_Ks;
    int m_Ds;
    vector<int> mark;
    vector<int> mark_w_iso;
    vector<bool> tree_mark;
    vector<int> offset; // offset of the first node of each component
    vector<vector<float> > dist_tables;

    uchar* vecs;    // database vectors
    int diff_argument = 1;
    // Building minimum spanning trees
    class Node {
    public:
        int id;
        int n_children=0;
        vector<Node*> children;
        float distance;    // distance to parent
        vector<tuple<uchar, uchar, uchar>> transform;// m, cid, cid
        //float* upper_bounds;
        float max_dist;     // upper bound of max distance of subtree
        Node();
        ~Node();
    };
    vector<Node*> trees;
    vector<vector<int>> tree_set;

    // query part
    double* MKKTable;
    float** m_sub_distances; // M x K, will change per query
    float** m_sub_diff_distances; // M x K x K, will change per query
    float** m_sub_bounds; // M x K, will change per query

    // This tree is rejected by Dong
    class Tree {
    public:
        int size;
        int* vec_ids;
        float* dists2c;     // distance to child
        // TODO how to represent diffs
        int* offsets;
        int* diff_offsets;
        uchar* root_vec;
        vector<tuple<uchar, uchar, uchar>> diffs;   // m, cid, cid

        float* max_dists;     // upper bound of max distance of its sub tree
    };
    Tree** diff_trees; 
    
    // Building MST
    uint* parents;
    uint* rank;
    uint max_tree_size;
    // Paper-Array's method
    struct NodeA{
        uint vec_id;
        uint parent_pos;
        uchar num_diff; // diffs from parent
    };
    NodeA* nodesa;
    // Paper-Queue's method
    struct NodeQ{
        uint vec_id;
        bool is_leaf;
        bool is_lastchild;
        uchar num_diff; // diffs from parent
    };
    NodeQ* nodesq;
    // Dong's method
    struct NodeD{
        uint vec_id;
        uint parent_pos;
        long long diffs_offset; // diffs from parent
        float max_dist;
        uint child_pos_start;
        bool is_last_child=false;
        int n_children;
    };
    uchar* root_codes;

    struct Diff{
        uchar m;
        uchar from;
        uchar to;
        Diff(){};
    };
    NodeD* nodes;
    Diff* diffs;
    long long n_diffs;
    long long n_trees;
    uint* tree_root_pos;    // mark the boundary of different 
                            // trees in the nodes array
    float* dist_array;      // For the online part
                            // distance is corresponding to the nodes array
    bool* is_last_child;    // 
    string dataset_path;

    // methods
    void create_dist_tables();
    float cal_distance_by_tables(int a, int b);
    void create_MKKTable();
    // Query part
    vector<pair<int, float> > Query(const vector<float> &query, int top_k); // for top-k search

    float cal_dist_bound(Node* node);
    float cal_dist_from_query(int id);
    void rotate_trees();
    void adjust_trees();
    static void write_groundtruth(std::string file_path, int M, int K, vector<vector<pair<uint,float>>>& results, int n_queries, int top_k);
    static void read_groundtruth(std::string file_path, int M, int K, vector<vector<pair<uint,float>>>& results, int n_queries, int top_k);
};

#endif
