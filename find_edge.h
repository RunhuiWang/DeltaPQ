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
#include "utils.h"
#include <parallel/algorithm>
#include <bitset>
using namespace std;

extern int PQ_M;
extern int PQ_K;

//#define MSTGENERATION 1

void nchoosek(int n, int k, vector<vector<int>> &combinations);

inline uint64_t GetCodeIndex(int M, uint64_t idx, int offset);

void print_code(unsigned char* codes, int M, int K, int LOG_K,int x);

void print_pair(unsigned char* codes, int M, int K, int LOG_K, int x, int y);

void partition(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees);
void partition_wang(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees);
void partition_new(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees);
void partition_linear(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees);
void partition_heap_hop(unsigned char* codes, int M, int K, int LOG_K, long long num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees);
inline void check_hash_code(int i1, int i3, const vector<pair<uint128_t, uint>> &hash_array, uint* parents, uint* rank, vector<vector<pair<int, float>>>& g_trees);
void partition_linear_opt(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<pair<uint, uint>>& edges, float sample_rate);
void partition_linear_opt(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees, float sample_rate);
void parallel_dist_diff_find_TA(unsigned char* codes, int M, int K, int LOG_K, uint num_codes, 
                int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
                vector<vector<pair<int, float>>>& g_trees, 
                vector<vector<float>>& dist_tables);
uint binary_search(const vector<pair<uint64_t, uint32_t>>& hash_array, 
                   uint64_t hashcode, uint start, uint end);
uint binary_search(uchar* codes, uint pos, uint end, uint128_t mask, int M);
uint binary_search(uchar* codes, vector<pair<uint, uint>>& pos_array, 
                    uint pos, uint end, uint128_t mask, int M);
inline uint128_t compute_value(uchar* code, uint128_t mask);

