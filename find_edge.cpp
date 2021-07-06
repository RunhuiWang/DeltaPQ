#include "find_edge.h"

#define UNDEFINED_DIFF_BY -1

extern int parallelism_enabled;
extern int NUM_THREAD;
//#define NUM_THREAD 20

//using namespace kgraph;

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

inline uint64_t GetCodeIndex(int M, uint64_t idx, int offset)
{
  return idx * M + offset;
}


void print_code(unsigned char* codes, int M, int K, int LOG_K,int x)
{
  cout << "code " << x << ": ";
  for (auto i = 0; i < M; i++)
    cout << (int)codes[GetCodeIndex(M, x, i)] << " ";
  cout << endl;
}

void print_pair(unsigned char* codes, int M, int K, int LOG_K,int x, int y)
{
  print_code(codes, M, K, LOG_K, x);
  print_code(codes, M, K, LOG_K, y);
  cout << endl;
}
void parallel_dist_diff_find_TA(unsigned char* codes, int M, int K, int LOG_K, uint num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            vector<vector<pair<int, float>>>& g_trees,
            vector<vector<float>>& dist_tables) 
{
    cout << "Find diff = " << DIFF_BY << " with distance" << endl;
    cout << get_current_time_str() << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    
    // sort the distances
    auto sorted_pairs = vector<vector<pair<double, int>>>(M);// pair.first is distance
                                                             // pair.second is c1*K+c2
    for (int m = 0; m < M; m++) {
        for (int c1 = 0; c1 < K; c1 ++) {
            for (int c2 = c1; c2 < K; c2 ++) {
                double dist = dist_tables[m][c1*K+c2];
                sorted_pairs[m].emplace_back(dist, c1*K+c2);// c1 <= c2
            }
        }
        sort(sorted_pairs[m].begin(), sorted_pairs[m].end(), 
            [](const pair<double,int>& a, const pair<double,int>& b) -> bool {
            return a.first < b.first;
        });
    }
    // comparison function for priority queue
    auto cmp = [](pair<double, pair<uint, uint>>& left, 
                  pair<double, pair<uint, uint>>& right) {
        return left.first < right.first;
    };
    // union find set to be used for cliques
    uint* clique_parents = new uint[num_codes];
    uint* clique_ranks = new uint[num_codes];
    vector<vector<pair<uint, uint>>> clique_trees;

    for (auto k = 0; k < combinations.size(); k++) {
        for (uint i = 0; i < num_codes; i ++) clique_parents[i] = i;
        for (uint i = 0; i < num_codes; i ++) clique_ranks[i] = 0;
            cout << k << " th combination start " << get_current_time_str() << endl;
        // calculate all hash values
        vector<pair<uint64_t, uint>> hash_array;
        hash_array.resize(num_codes);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint64_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
            hash_array[l] = make_pair(hash, l);
        }
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(), 
            [](const pair<uint64_t, uint32_t>& a, const pair<uint64_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        // set up an offset array (get cliques)
        vector<uint> offsets;
        offsets.push_back(0);
        for (uint i = 1; i < num_codes; i ++) {
            if (hash_array[i].first != hash_array[i-1].first) {
                offsets.push_back(i);
            }
        }
        offsets.push_back(num_codes);
        clique_trees.resize(offsets.size()-1);
        cout << clique_trees.size() << " cliques" << endl;
        // DEBUG INFO
        int nthreads = omp_get_thread_num()+1;
        cout << "nthreads " << nthreads << endl;
        nthreads = 100;
        vector<long long> debug_n_pairs_checked(nthreads);
        vector<long long> debug_max_pairs_in_clique(nthreads);
        vector<long long> debug_max_clique_size(nthreads);
        // END OF DEBUG INFO
        // check each clique in parallel 
        #pragma omp parallel for
        for (uint i = 0; i < offsets.size()-1; i ++) {
            clique_trees[i].resize(0);  // make sure the edge list is empty
            // use TA algorithm to find top n-1 distances
            uint pair_start = offsets[i], pair_end = offsets[i+1];
            if (pair_start + 1 == pair_end) continue;
            vector<int> sub_dims;   // get the dimensions that are not used for hashing
            vector<bool> marks(M);  // mark true if a sub dimension exists in combination
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                marks[*it] = true;
            for (int m = 0; m < M; m ++) 
                if (marks[m] == false) sub_dims.push_back(m);
            // now we have found the dimensions for TA algorithm
            // create an inverted list for each sub dimension
            auto inverted_lists = 
                    vector<vector<vector<uint>>>(M, vector<vector<uint>>(K));
            // traverse all the vecs to create the lists
            for (uint j = pair_start; j < pair_end; j ++) {
                uint vec_id = hash_array[j].second;
                for (int m : sub_dims) {
                    int cid = (int)codes[vec_id*M+m];
                    inverted_lists[m][cid].emplace_back(vec_id);
                }
            }
            // now we have sorted distances and inverted lists, perform TA algo
            priority_queue<pair<double, pair<uint,uint>>, 
                            vector<pair<double, pair<uint, uint>>>, 
                                        decltype(cmp)> pq(cmp); // a max heap
            unordered_set<pair<uint, uint>, pair_hash> checked_pairs;    
                                                // pair_hash defined in utils.h
            uint n_edges_found = 0;
            uint n_target_edges = pair_end - pair_start-1;
            
            int thread_id = omp_get_thread_num();   // DEBUG
            long long debug_pairs_checked = 0;
            for (int cursor = 0; cursor < sorted_pairs[0].size(); cursor ++) {
                // get all the node pairs
                for (auto m : sub_dims) {
                    int index = sorted_pairs[m][cursor].second;
                    int c1 = index / K, c2 = index % K; // c1 <= c2
                    // get all node pairs by a Cartesian product
                    for (auto vec1 : inverted_lists[m][c1]) {
                        for (auto vec2: inverted_lists[m][c2]) {
                            if (vec1 == vec2) continue;
                            auto pair = make_pair(vec1, vec2);
                            debug_n_pairs_checked[thread_id]++;// DEBUG
                            debug_pairs_checked ++;
                            if (checked_pairs.insert(pair).second) {
                                // this pair has not been checked before
                                double edge_dist = 0;
                                for (auto mm:sub_dims) {
                                    int cid1 = (int)codes[vec1*M+mm];
                                    int cid2 = (int)codes[vec2*M+mm];
                                    edge_dist += dist_tables[mm][cid1*K+cid2];
                                }
                                pq.push(make_pair(edge_dist, pair));
                            }
                        }
                    }
                }
                // check the maximum sum of distances from next level
                double threshold = 0;
                for (auto m : sub_dims) {
                    threshold += sorted_pairs[m][cursor+1].first;
                }
                // check each pair, if the distatnce is smaller than threshold,
                // add an edge into the union_find set
                while(!pq.empty() && pq.top().first <= threshold) {
                    auto pair = pq.top().second;
                    pq.pop();
                    // do a read only find_parent here
                    uint vec1 = pair.first;
                    uint vec2 = pair.second;
                    uint p1 = find_set(clique_parents, vec1);
                    uint p2 = find_set(clique_parents, vec2);
                    //cout << vec1 << " " << vec2 << endl;
                    if (p1 != p2) {
                        if (clique_ranks[p1] > clique_ranks[p2]) clique_parents[p2] = p1;
                        else clique_parents[p1] = p2;
                        if (clique_ranks[p1] == clique_ranks[p2]) clique_ranks[p2] ++;
                        clique_trees[i].emplace_back(vec1, vec2);
                        n_edges_found ++;
                    }
                }
                if (n_edges_found == n_target_edges) break;
            }
            //cout << n_edges_found << endl;
            if (debug_pairs_checked > debug_max_pairs_in_clique[thread_id]) {
                debug_max_pairs_in_clique[thread_id] = debug_pairs_checked;
                debug_max_clique_size[thread_id] = pair_end - pair_start;
            }
            
        }
        uint n_edges = 0;
        // put the collected edges from cluques to the global trees
        for (uint i = 0; i < offsets.size()-1; i ++) {
            n_edges += clique_trees[i].size();
            for (auto& edge : clique_trees[i]) {
                uint vec1 = edge.first;
                uint vec2 = edge.second;
                uint p1 = find_set(parents, vec1);
                uint p2 = find_set(parents, vec2);
                if (p1 != p2) {
                    if (rank[p1] > rank[p2]) parents[p2] = p1;
                    else parents[p1] = p2;
                    if (rank[p1] == rank[p2]) rank[p2] ++;
                    g_trees[vec1].push_back(make_pair(vec2,UNDEFINED_DIFF_BY));
                    g_trees[vec2].push_back(make_pair(vec1,UNDEFINED_DIFF_BY));
                }
            }
        }
        long long debug_total_pairs_checked = 0;
        long long debug_max_pairs = 0;
        cout << "number of pairs checked in each thread: " << endl;
        for (int i = 0; i < nthreads; i ++) {
            cout << debug_n_pairs_checked[i] << " ";
            if (debug_n_pairs_checked[i] > debug_max_pairs)
                debug_max_pairs = debug_n_pairs_checked[i];
        }
        cout << endl;
        cout << "Total number of pairs checked " << debug_total_pairs_checked << endl;
        cout << "Max pairs processed in a thread" << debug_max_pairs << endl;
        long long max_pairs_in_clique = 0;
        long long max_clique = 0;
        cout << "max number of pairs checked in one clique: " << endl;
        for (int i = 0; i < nthreads; i ++) {
            cout <<  debug_max_pairs_in_clique[i]<< ",";
            cout <<  debug_max_clique_size[i] << " ";
            if (debug_max_pairs_in_clique[i] > max_pairs_in_clique) {
                max_pairs_in_clique = debug_max_pairs_in_clique[i];
                max_clique = debug_max_clique_size[i];
            }
        }
        cout << endl;
        cout << "max pairs in one clique " << max_pairs_in_clique << endl;
        cout << "max clique size         " << max_clique << endl;
        cout << "Number of edges inserted " << n_edges << endl;
        cout << "+++++++++++++++++++++" << endl;
    }
    delete[] clique_parents;
    delete[] clique_ranks;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}
void print_code(const unsigned char* codes, int M, long long codeid)
{
    for (auto prt2 = 0; prt2 < M; prt2++)
        cerr << (int)codes[codeid * M + prt2] << " ";
    cerr << endl;
}

/*
void partition_heap_hop(unsigned char* codes, int M, int K, int LOG_K, long long num_codes, 
        int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
        atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{

    if (DIFF_BY < 2)
    {
        partition_linear_opt(codes, M, K, LOG_K, num_codes, PART_NUM, DIFF_BY, 
                parents, rank, lock, g_trees);
        return;
    }

    int diff = DIFF_BY;
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    assert(DIFF_BY >= 2);

    nchoosek(M, M - DIFF_BY, combinations);

    /////// get the dimensions that are not blackouted!
    vector<vector<bool>> blackouts(combinations.size(), vector<bool>(M, true));
    for (int comb_id = 0; comb_id < combinations.size(); comb_id++)
    {
        // mark true if a sub dimension exists in combination and should be blackout
        for (auto it = combinations[comb_id].begin(); it != combinations[comb_id].end(); it++)
            blackouts[comb_id][*it] = false;
    }

    vector<vector<int>> blackouts_dims(combinations.size());
    for (int comb_id = 0; comb_id < combinations.size(); comb_id++)
    {
        for (auto i = 0; i < M; i++)
            if (blackouts[comb_id][i])
                blackouts_dims[comb_id].push_back(i);
    }

    //// get execution plan
    vector<map<int, vector<int>>> execution_plan(M); // order by first dimension, last diff, combination id
    for (int i = 0; i < combinations.size(); i++)
    {
        auto &blkout = blackouts_dims[i];
        int max_sort_from = blkout.front();
        int max_last_diff = blkout.back();
        int max_gap = (M - blkout.back()) + blkout.front();  // sort from the last dim (include first, not include the last)
        for (int j = 0; j < ((int)blkout.size()) - 1; j++)
        {
            int gap = blkout[j + 1] - blkout[j];
            if (gap > max_gap)
            {
                max_sort_from = blkout[j + 1];
                max_last_diff = blkout[j];
                max_gap = gap;
            }
        }
        execution_plan[max_sort_from][max_last_diff].push_back(i);
    }

    for (auto i = 0; i < execution_plan.size(); i++)
    {
        cout << "====== NEW Sort from " << endl;
        for (auto &entry : execution_plan[i])
        {
            cout << "New Last Diff. Num: " << entry.second.size() << endl;
            for (auto id : entry.second)
            {
                cout << "sort from: " << i;
                cout << "  last diff is: " << entry.first;
                cout << "  blackouts: ";
                for (auto x : blackouts_dims[id])
                    cout << x << " ";
                cout << endl;
            }
        }
        cout << endl << endl;
    }
    /// got execution plan

    // Preparing
    uint code_id_sorted[num_codes];
    for (auto i = 0; i < num_codes; i++)
        code_id_sorted[i] = i;

    int kth_comb = 0;

    for (int sort_from = M - 1; sort_from > 0; sort_from--)
    {

        // ****  Sort All Codes from sort_from dim ****
        // ascending lexicographical order from sort_from
        sort(code_id_sorted, code_id_sorted + num_codes, [codes, M, sort_from](const uint a, const uint b) {
                for (int i = 0; i < PQ_M; i++)
                {
                // ring compare
                if (codes[((long long)a) * M + (i + sort_from) % M] <
                    codes[((long long)b) * M + (i + sort_from) % M])
                return true;
                if (codes[((long long)a) * M + (i + sort_from) % M] >
                    codes[((long long)b) * M + (i + sort_from) % M])
                return false;
                }
                return false;
                });

        ////// printing
        
        cerr << "SORT FROM: " << sort_from << endl;
        cerr << " ascending lexicographical order from sort_from: " << endl;
        for (auto prt = 0; prt < 50; prt++)
        {
            cerr << code_id_sorted[prt] << " ";
            print_code(codes, M, code_id_sorted[prt]);
        }

        for (auto &entry : execution_plan[sort_from])
        {
            if (num_codes >= 1000000000)
                cout << kth_comb << " th combination start " << get_current_time_str() << endl;
            // cout << "   find edges " << get_current_time_str() << endl;

            // get the last difference dimension
            int last_diff = entry.first;
            // cerr << "last diff: " << last_diff << endl;

            /////// Given any two integers X and Y within [0, M);
            /////// In a ring, the distance from X to Y (include both) is:
            ///////  Y - X + 1     (if Y >= X) 
            ///////  Y + M - X + 1 (if Y <  X)
            ///////  In all cases, they are 
            ///////  the lexicographical order is
            ///////  X%M, (X+1)%M, (X+2)%M, ....., (X+dis-1)%M == Y
            uint pos = 0;
            vector<pair<uint,uint>> pos_array;
            do {
                auto ptr_next_pos = upper_bound(code_id_sorted + pos, code_id_sorted + num_codes, code_id_sorted[pos], [last_diff, sort_from, M, codes](const uint a, const uint b) {
                        //// com pare from sort_from to last dim (included, without blackout)
                        // X is sort_from
                        // Y is last_diff
                        int dis = last_diff - sort_from + 1;
                        if (last_diff < sort_from)
                        dis = last_diff + M - sort_from + 1;

                        // a and b are code id in code_id_sorted array
                        // wanted to compare these two codes
                        for (int i = 0; i < dis; i++)
                        {
                        // ring compare
                        if (codes[((long long)a) * M + (i + sort_from) % M] <
                            codes[((long long)b) * M + (i + sort_from) % M])
                        return true;
                        if (codes[((long long)a) * M + (i + sort_from) % M] >
                            codes[((long long)b) * M + (i + sort_from) % M])
                        return false;
                        }
                        return false;
                });

                int advance_steps = distance(code_id_sorted + pos, ptr_next_pos);
                // cerr << "next pos in pas_array " << pos << endl;
                pos_array.emplace_back(pos, pos + advance_steps);
                pos += advance_steps;
            } while (pos < num_codes);

            cerr << "pos arrary size: " << pos_array.size() << endl;
            cerr << "print 10 array ORDER 1 NO BLACKOUT:" << endl;
            cerr << "  should sorted by sorted_from to last diff (all includes): " << sort_from << " " << last_diff << endl;
            for (auto debug_id = 0; debug_id < 10; debug_id++)
            {
                cerr << code_id_sorted[pos_array[debug_id].first] << ":  **" << pos_array[debug_id].first << "**";
                print_code(codes, M, code_id_sorted[pos_array[debug_id].first]);
            }

            for (auto comb_id : entry.second) 
            {
                cout << "combinations: ";
                for (auto debug_id : combinations[comb_id])
                    cerr << debug_id << " ";
                cerr << endl;
                kth_comb++;
                ///// just run heap_skip on all of them
                ///// OPT: sort and heap_skip on each of them
                ///// copy_array is the heap
                auto copy_array = pos_array;  ///// DO NOT TOUCH POS_ARRAY!!

                auto comp_heap = [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted](const pair<uint, uint> &a, const pair<uint, uint> &b) {
                    //// should return a > b for min-heap
                    ///// a.first is the pointer
                    ///// code_id_sorted[pointer] is the real code_id
                    ///// code[code_id * M] is the code
                    ///// wanted to return a's code > b's code in the following order 
                    ///// almost same order with binary search above when building the copy_array
                    ///// from sort_from+1 to sort_from-1 (in a ring) total M-1

                    assert(blackouts[comb_id][sort_from]);

//cerr << "cmp" << endl;
//print_code(codes, M, code_id_sorted[a.first]);
//print_code(codes, M, code_id_sorted[b.first]);
                    for (int i = 0; i < M - 1; i++)
                    {
                        ///// (sort_from+1)%M to (sort_from+M-1)%M)
                        ///// BLACK OUT, DO NOT FORGET
                        int idx = (sort_from + 1 + i) % M;
                        if (blackouts[comb_id][idx]) continue;
//cerr << idx << endl;
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] >
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
{
//cerr << "cm he true" <<endl;
                            return true;
}
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] <
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
{
//cerr << "cm he false" <<endl;
                            return false;
}
                    }
//cerr << "cm he false" <<endl;
                    return false;
                };

                auto is_equal_heap = [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted](const pair<uint, uint> &a, const pair<uint, uint> &b) {
                    for (int i = 0; i < M - 1; i++)
                    {
                        int idx = (sort_from + 1 + i) % M;
                        if (blackouts[comb_id][idx]) continue;
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] !=
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
                            return false;
                    }
                    return true;
                };

                cerr << "make_heap start" << endl;
                make_heap(copy_array.begin(), copy_array.end(), comp_heap);
                
                cerr << "make_heap end; at the top" << endl;
                for (int debug_m = 0; debug_m < M; debug_m++)
                    cerr << (int)codes[(long long)code_id_sorted[copy_array.front().first] * M + debug_m] << " ";
                cerr << endl;

                int heap_size = copy_array.size();
                vector<int> clique;
                vector<vector<int>> cliques;
                while (heap_size > 1)
                {
                    pop_heap(copy_array.begin(), copy_array.end(), comp_heap);
                    --heap_size;
                    ///// HERE I want to put in the id of code into clique
                    ///// copy_array[heap_size] is what i just popped out
                    ///// popped_out.first is the pointer
                    ///// code_id_sorted[pointer] is the real code_id
                    clique.push_back(code_id_sorted[copy_array[heap_size].first]);
                    
                    /// debug
                    
                    cerr << "popped out ";
                    for (int debug_m = 0; debug_m < M; debug_m++)
                        cerr << (int)codes[(long long)code_id_sorted[copy_array[heap_size].first] * M + debug_m] << " ";
                    cerr << endl;

                    ///// AT TOP IS copy_array.front()
                    ///// just popped out is copy_array[heap_size]
                    if (!is_equal_heap(copy_array.front(), copy_array[heap_size]))
                    {
                        if (clique.size() > 1)
                            cliques.push_back(std::move(clique));
                        clique.clear();
                    }


                      cerr << endl << "=========start====" << endl;
                      cerr << "blackout" << endl;
                      for (int debug_m = 0; debug_m < M; debug_m++)
                        if (blackouts[comb_id][debug_m])
                          cerr << debug_m << " true" << endl;
                        else
                          cerr << debug_m << " fasle" << endl;

                      cerr << "sort_from: " << sort_from << endl;
                      cerr << "=========CURRENT AT THE TOP ====" << copy_array.front().first << " " << copy_array.front().second << endl;
                      for (int debug_m = 0; debug_m < M; debug_m++)
                        cerr << (int)codes[((long long)code_id_sorted[copy_array.front().first]) * M + debug_m] << " ";
                      cerr << endl;
                      cerr << "===== JUST POPPED ===" << copy_array[heap_size].first << " " << copy_array[heap_size].second << " " << heap_size << endl;
                      for (int debug_id = copy_array[heap_size].first; debug_id < copy_array[heap_size].second; debug_id++)
                      {
                        cerr << debug_id << " print ";
                        for (int debug_m = 0; debug_m < M; debug_m++)
                          cerr << (int)codes[((long long)code_id_sorted[debug_id]) * M + debug_m] << " ";
                        cerr << endl;
                      }


                    // BINARY_MOVE_TO_FIRST_LARGER_THAN_AT_TOP;
                    // we are searching in the code_id_sorted array
                    // we are moving this pointer: copy_array[heap_size].first
                    // the pointer points to the code_id code_id_sorted[pointer]
                    // the code is codes[code_id * M]
                    //
                    // we want to move to the first larger than the one at top
                    // which is the code pointed by copy_array.front().first
                    // the code id is code_id_sorted[copy_array.front().first]
                    // the code is  codes[code_id * M]
                    auto ptr_first_gt_pos = upper_bound(code_id_sorted + copy_array[heap_size].first,
                            code_id_sorted + copy_array[heap_size].second,
                            code_id_sorted[copy_array.front().first], [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted] (const uint a, const uint b) {
                            /// a and b are pointers to code_id_sorted
                            /// their order is determined by the reverse order of in comp_heap
                            assert(blackouts[comb_id][sort_from]);
                            cerr << " comparing " << a << " v " << b;
                            for (int i = 0; i < M; i++)
                                cerr << (int) codes[((long long)a) * M + i] << " ";
                                cerr << endl;
                            for (int i = 0; i < M; i++)
                                cerr << (int) codes[((long long)b) * M + i] << " ";
                                cerr << endl;

                            for (int i = 0; i < M - 1; i++)
                            {
                            int idx = (sort_from + 1 + i) % M;
                            if (blackouts[comb_id][idx]) continue;
                            if (codes[((long long)a) * M + idx] <
                                codes[((long long)b) * M + idx])
{
cerr << "return true" << endl;
                            return true;
}
                            if (codes[((long long)a) * M + idx] >
                                codes[((long long)b) * M + idx])
{
cerr << "return false" << endl;
                            return false;
}
                            }
cerr << "return false" << endl;
                            return false;
                            });
                    

                    int jump_dis = distance(code_id_sorted + copy_array[heap_size].first, ptr_first_gt_pos);
                    cerr << "adv: " << jump_dis << " from: " << copy_array[heap_size].first << " to " << copy_array[heap_size].second << " asize: " << copy_array.size() << endl;

                    copy_array[heap_size].first += jump_dis;

                    //// Test if reach the end for the just popped one
                    if (copy_array[heap_size].first == copy_array[heap_size].second)
                        copy_array.pop_back();
                    else {
                        push_heap(copy_array.begin(), copy_array.end(), comp_heap);
                        ++heap_size;
                    }
                }
                 
                if (clique.size() > 0)
                {
                    clique.push_back(code_id_sorted[copy_array.front().first]);
                    cliques.push_back(std::move(clique));
                }


                for (auto &aclique : cliques)
                {
                    cout << "CLIQUE SIZE: " << aclique.size() << endl;
                    for (uint j = 1; j < aclique.size(); j++) 
                    {
                        uint start_i = aclique[j-1];
                        uint end_i = aclique[j];
                        uint x = find_set(parents, start_i);
                        uint y = find_set(parents, end_i);
                        //if (s_id >= e_id) continue;
                        if (x != y) {
                            // add this edge into the tree
                            if (rank[x] > rank[y]) parents[y] = x;
                            else parents[x] = y;
                            if (rank[x] == rank[y]) rank[y]++;

                            cerr << "ADDDDEDDD " << endl;
                            g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                            g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                        }
                    }
                }

                // cout << "   find edges done" << get_current_time_str() << endl;
            }
        }
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}
debug
*/

void partition_heap_hop(unsigned char* codes, int M, int K, int LOG_K, long long num_codes, 
        int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
        atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{

    if (DIFF_BY < 2)
    {
        partition_linear_opt(codes, M, K, LOG_K, num_codes, PART_NUM, DIFF_BY, 
                parents, rank, lock, g_trees, 0.5);
        return;
    }

    int diff = DIFF_BY;
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    assert(DIFF_BY >= 2);

    nchoosek(M, M - DIFF_BY, combinations);

    /////// get the dimensions that are not blackouted!
    vector<vector<bool>> blackouts(combinations.size(), vector<bool>(M, true));
    for (int comb_id = 0; comb_id < combinations.size(); comb_id++)
    {
        // mark true if a sub dimension exists in combination and should be blackout
        for (auto it = combinations[comb_id].begin(); it != combinations[comb_id].end(); it++)
            blackouts[comb_id][*it] = false;
    }

    vector<vector<int>> blackouts_dims(combinations.size());
    for (int comb_id = 0; comb_id < combinations.size(); comb_id++)
    {
        for (auto i = 0; i < M; i++)
            if (blackouts[comb_id][i])
                blackouts_dims[comb_id].push_back(i);
    }

    //// get execution plan
    vector<map<int, vector<int>>> execution_plan(M); // order by first dimension, last diff, combination id
    for (int i = 0; i < combinations.size(); i++)
    {
        auto &blkout = blackouts_dims[i];
        int max_sort_from = blkout.front();
        int max_last_diff = blkout.back();
        int max_gap = (M - blkout.back()) + blkout.front();  // sort from the last dim (include first, not include the last)
        for (int j = 0; j < ((int)blkout.size()) - 1; j++)
        {
            int gap = blkout[j + 1] - blkout[j];
            if (gap > max_gap)
            {
                max_sort_from = blkout[j + 1];
                max_last_diff = blkout[j];
                max_gap = gap;
            }
        }
        execution_plan[max_sort_from][max_last_diff].push_back(i);
    }

    /*
    for (auto i = 0; i < execution_plan.size(); i++)
    {
        cout << "====== NEW Sort from " << endl;
        for (auto &entry : execution_plan[i])
        {
            cout << "New Last Diff. Num: " << entry.second.size() << endl;
            for (auto id : entry.second)
            {
                cout << "sort from: " << i;
                cout << "  last diff is: " << entry.first;
                cout << "  blackouts: ";
                for (auto x : blackouts_dims[id])
                    cout << x << " ";
                cout << endl;
            }
        }
        cout << endl << endl;
    }
    */
    /// got execution plan

    // Preparing
    uint code_id_sorted[num_codes];
    for (auto i = 0; i < num_codes; i++)
        code_id_sorted[i] = i;

    int kth_comb = 0;

    for (int sort_from = 0; sort_from < M; sort_from++)
    {

        // ****  Sort All Codes from sort_from dim ****
        // ascending lexicographical order from sort_from
        sort(code_id_sorted, code_id_sorted + num_codes, [codes, M, sort_from](const uint a, const uint b) {
                for (int i = 0; i < PQ_M; i++)
                {
                // ring compare
                if (codes[((long long)a) * M + (i + sort_from) % M] <
                    codes[((long long)b) * M + (i + sort_from) % M])
                return true;
                if (codes[((long long)a) * M + (i + sort_from) % M] >
                    codes[((long long)b) * M + (i + sort_from) % M])
                return false;
                }
                return false;
                });

        for (auto &entry : execution_plan[sort_from])
        {
            if (num_codes >= 1000000000)
                cout << kth_comb << " th combination start " << get_current_time_str() << endl;
            // cout << "   find edges " << get_current_time_str() << endl;

            // get the last difference dimension
            int last_diff = entry.first;
            // cerr << "last diff: " << last_diff << endl;

            /////// Given any two integers X and Y within [0, M);
            /////// In a ring, the distance from X to Y (include both) is:
            ///////  Y - X + 1     (if Y >= X) 
            ///////  Y + M - X + 1 (if Y <  X)
            ///////  In all cases, they are 
            ///////  the lexicographical order is
            ///////  X%M, (X+1)%M, (X+2)%M, ....., (X+dis-1)%M == Y
            uint pos = 0;
            vector<pair<uint,uint>> pos_array;
            do {
                auto ptr_next_pos = upper_bound(code_id_sorted + pos, code_id_sorted + num_codes, code_id_sorted[pos], [last_diff, sort_from, M, codes](const uint a, const uint b) {
                        //// com pare from sort_from to last dim (included, without blackout)
                        // X is sort_from
                        // Y is last_diff
                        int dis = last_diff - sort_from + 1;
                        if (last_diff < sort_from)
                        dis = last_diff + M - sort_from + 1;

                        // a and b are code id in code_id_sorted array
                        // wanted to compare these two codes
                        for (int i = 0; i < dis; i++)
                        {
                        // ring compare
                        if (codes[((long long)a) * M + (i + sort_from) % M] <
                            codes[((long long)b) * M + (i + sort_from) % M])
                        return true;
                        if (codes[((long long)a) * M + (i + sort_from) % M] >
                            codes[((long long)b) * M + (i + sort_from) % M])
                        return false;
                        }
                        return false;
                });

                int advance_steps = distance(code_id_sorted + pos, ptr_next_pos);
                // cerr << "next pos in pas_array " << pos << endl;
                pos_array.emplace_back(pos, pos + advance_steps);
                pos += advance_steps;
            } while (pos < num_codes);

            cerr << "pos arrary size: " << pos_array.size() << endl;

            for (auto comb_id : entry.second) 
            {
                cout << "combinations: ";
                for (auto debug_id : combinations[comb_id])
                    cerr << debug_id << " ";
                cerr << endl;
                kth_comb++;
                ///// just run heap_skip on all of them
                ///// OPT: sort and heap_skip on each of them
                ///// copy_array is the heap
                auto copy_array = pos_array;  ///// DO NOT TOUCH POS_ARRAY!!

                auto comp_heap = [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted](const pair<uint, uint> &a, const pair<uint, uint> &b) {
                    //// should return a > b for min-heap
                    ///// a.first is the pointer
                    ///// code_id_sorted[pointer] is the real code_id
                    ///// code[code_id * M] is the code
                    ///// wanted to return a's code > b's code in the following order 
                    ///// almost same order with binary search above when building the copy_array
                    ///// from sort_from+1 to sort_from-1 (in a ring) total M-1

                    assert(blackouts[comb_id][sort_from]);

                    for (int i = 0; i < M - 1; i++)
                    {
                        ///// (sort_from+1)%M to (sort_from+M-1)%M)
                        ///// BLACK OUT, DO NOT FORGET
                        int idx = (sort_from + 1 + i) % M;
                        if (blackouts[comb_id][idx]) continue;
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] >
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
                            return true;
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] <
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
                            return false;
                    }
                    return false;
                };

                auto is_equal_heap = [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted](const pair<uint, uint> &a, const pair<uint, uint> &b) {
                    for (int i = 0; i < M - 1; i++)
                    {
                        int idx = (sort_from + 1 + i) % M;
                        if (blackouts[comb_id][idx]) continue;
                        if (codes[((long long)code_id_sorted[a.first]) * M + idx] !=
                                codes[((long long)code_id_sorted[b.first]) * M + idx])
                            return false;
                    }
                    return true;
                };

                make_heap(copy_array.begin(), copy_array.end(), comp_heap);
                
                int heap_size = copy_array.size();
                vector<int> clique;
                vector<vector<int>> cliques;
                while (heap_size > 1)
                {
                    pop_heap(copy_array.begin(), copy_array.end(), comp_heap);
                    --heap_size;
                    ///// HERE I want to put in the id of code into clique
                    ///// copy_array[heap_size] is what i just popped out
                    ///// popped_out.first is the pointer
                    ///// code_id_sorted[pointer] is the real code_id
                    clique.push_back(code_id_sorted[copy_array[heap_size].first]);
                    
                    ///// AT TOP IS copy_array.front()
                    ///// just popped out is copy_array[heap_size]
                    if (!is_equal_heap(copy_array.front(), copy_array[heap_size]))
                    {
                        if (clique.size() > 1)
                            cliques.push_back(std::move(clique));
                        clique.clear();
                    }
                    // BINARY_MOVE_TO_FIRST_LARGER_THAN_AT_TOP;
                    // we are searching in the code_id_sorted array
                    // we are moving this pointer: copy_array[heap_size].first
                    // the pointer points to the code_id code_id_sorted[pointer]
                    // the code is codes[code_id * M]
                    //
                    // we want to move to the first larger than the one at top
                    // which is the code pointed by copy_array.front().first
                    // the code id is code_id_sorted[copy_array.front().first]
                    // the code is  codes[code_id * M]
                    auto ptr_first_gt_pos = upper_bound(code_id_sorted + copy_array[heap_size].first,
                            code_id_sorted + copy_array[heap_size].second,
                            code_id_sorted[copy_array.front().first], [&blackouts, &codes, comb_id, sort_from, M, &code_id_sorted] (const uint a, const uint b) {
                            /// a and b are pointers to code_id_sorted
                            /// their order is determined by the reverse order of in comp_heap
                            assert(blackouts[comb_id][sort_from]);

                            for (int i = 0; i < M - 1; i++)
                            {
                            int idx = (sort_from + 1 + i) % M;
                            if (blackouts[comb_id][idx]) continue;
                            if (codes[((long long)a) * M + idx] <
                                codes[((long long)b) * M + idx])
                            return true;
                            if (codes[((long long)a) * M + idx] >
                                codes[((long long)b) * M + idx])
                            return false;
                            }
                            return false;
                            });
                    

                    int jump_dis = distance(code_id_sorted + copy_array[heap_size].first, ptr_first_gt_pos);
                    copy_array[heap_size].first += jump_dis;

                    //// Test if reach the end for the just popped one
                    if (copy_array[heap_size].first == copy_array[heap_size].second)
                        copy_array.pop_back();
                    else {
                        push_heap(copy_array.begin(), copy_array.end(), comp_heap);
                        ++heap_size;
                    }
                }
                 
                if (clique.size() > 0)
                {
                    clique.push_back(code_id_sorted[copy_array.front().first]);
                    cliques.push_back(std::move(clique));
                }


                for (auto &aclique : cliques)
                {
                    for (uint j = 1; j < aclique.size(); j++) 
                    {
                        uint start_i = aclique[j-1];
                        uint end_i = aclique[j];
                        uint x = find_set(parents, start_i);
                        uint y = find_set(parents, end_i);
                        //if (s_id >= e_id) continue;
                        if (x != y) {
                            // add this edge into the tree
                            if (rank[x] > rank[y]) parents[y] = x;
                            else parents[x] = y;
                            if (rank[x] == rank[y]) rank[y]++;

                            g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                            g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                        }
                    }
                }

                // cout << "   find edges done" << get_current_time_str() << endl;
            }
        }
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}

uint binary_search(uchar* codes, uint pos, uint end, uint128_t mask, int M) {
    if (end - pos == 1) return end;
    uint128_t val0 = compute_value(codes+(long long)pos*M, mask);
    uint mid = (pos+end) / 2;
    uint128_t val_mid = compute_value(codes+(long long)mid*M, mask);
    if (val_mid > val0) {
        return binary_search(codes, pos, mid, mask, M);
    } else {
        return binary_search(codes, mid, end, mask, M);
    }
}
uint binary_search(uchar* codes, vector<pair<uint, uint>>& pos_array,uint pos,
                    uint end, uint128_t mask, int M) {
    if (end - pos == 1) return end;
    uint128_t val0 = compute_value(codes+(long long)(pos_array[pos].first)*M, mask);
    uint mid = (pos+end) / 2;
    uint128_t val_mid = compute_value(codes+(long long)(pos_array[pos].first)*M, mask);
    if (val_mid > val0) {
        return binary_search(codes, pos_array, pos, mid, mask, M);
    } else {
        return binary_search(codes, pos_array, mid, end, mask, M);
    }

}
inline uint128_t compute_value(uchar* code, uint128_t mask) {
    uint128_t value = *((uint128_t*)code);
    return value & mask;
}


inline void check_hash_code(int i1, int i3, const vector<pair<uint128_t, uint>> &hash_array, uint* parents, uint* rank, vector<vector<pair<int, float>>>& g_trees)
{
  if (i1 == i3)
  {
    if (hash_array[i1].first == hash_array[i1 - 1].first)
    {
      uint start_i = hash_array[i1].second;
      uint end_i = hash_array[i1-1].second;
      uint x = find_set(parents, start_i);
      uint y = find_set(parents,end_i);
      if (x != y) {
        if (rank[x] > rank[y]) parents[y] = x;
        else parents[x] = y;
        if (rank[x] == rank[y]) rank[y] ++;
        g_trees[start_i].push_back(make_pair(end_i,.0));
        g_trees[end_i].push_back(make_pair(start_i,.0));
      }
    }
  }
  else if (hash_array[i1].first == hash_array[i3].first)
  {
    if (hash_array[i1].first == hash_array[i1 - 1].first)
    {
      uint start_i = hash_array[i1].second;
      uint end_i = hash_array[i1-1].second;
      uint x = find_set(parents, start_i);
      uint y = find_set(parents,end_i);
      if (x != y) {
        if (rank[x] > rank[y]) parents[y] = x;
        else parents[x] = y;
        if (rank[x] == rank[y]) rank[y] ++;
        g_trees[start_i].push_back(make_pair(end_i,.0));
        g_trees[end_i].push_back(make_pair(start_i,.0));
      }
    }
    for (uint j = i1; j < i3; j++) {

      uint start_i = hash_array[j].second;
      uint end_i = hash_array[j+1].second;
      uint x = find_set(parents, start_i);
      uint y = find_set(parents,end_i);
      if (x != y) {
        if (rank[x] > rank[y]) parents[y] = x;
        else parents[x] = y;
        if (rank[x] == rank[y]) rank[y] ++;
        g_trees[start_i].push_back(make_pair(end_i,.0));
        g_trees[end_i].push_back(make_pair(start_i,.0));
      }
    }
  }
  else
  {
    int i2 = (i1 + i3) / 2;
    check_hash_code(i1, i2, hash_array, parents, rank, g_trees);
    check_hash_code(i2 + 1, i3, hash_array, parents, rank, g_trees);
  }
}
 
 /*
void partition_linear_opt(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees, float sample_rate)
{
    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    //assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;
    if (M>8) {
        random_shuffle(combinations.begin(), combinations.end());
        combinations.resize(round(combinations.size() * sample_rate));
    }
    vector<pair<uint128_t, uint>> hash_array;
    hash_array.resize(num_codes);
    cout << combinations.size() << " combinationos" << endl;
    for (auto k = 0; k < combinations.size(); k++) {
        if (num_codes >= 1000000000)
            cout << k << " th combination start " << get_current_time_str() << endl;
        // calculate all hash values
        //vector<pair<uint64_t, uint>> hash_array;
        //cout << "   resize hash array " << get_current_time_str() << endl;
        //cout << "   calculate hash code " << get_current_time_str() << endl;

        gettimeofday(&beg, NULL);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint128_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
            hash_array[l] = make_pair(hash, l);
        }
        gettimeofday(&mid, NULL);
        
        cout << "   calculate hash codes  " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        gettimeofday(&beg, NULL);
        // sort the hash codes
        // Explicitly force a call to parallel sort.
        __gnu_parallel::sort(hash_array.begin(), hash_array.end(), [](const pair<uint128_t, uint32_t>& a, const pair<uint128_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        gettimeofday(&mid, NULL);
        cout << "   sort codes " << mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
        gettimeofday(&beg, NULL);
        // traverse hash array
        

#ifdef MSTGENERATION
        for (uint i = 0; i < num_codes; i ++) {
            uint end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            // check each pair
            for (uint j = i; j < end-1; j ++) {
                for (uint k = j + 1; k < end-1; k ++) {
                    uint start_i = hash_array[j].second;
                    uint end_i = hash_array[k].second;
                    uint x = find_set(parents, start_i);
                    uint y = find_set(parents,end_i);
                    //if (s_id >= e_id) continue;
                    if (x != y) {
                        // add this edge into the tree
                        if (rank[x] > rank[y]) parents[y] = x;
                        else parents[x] = y;
                        if (rank[x] == rank[y]) rank[y] ++;

                        g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                        g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                    }
                }
            }
            i = end - 1;
        }
#else
        check_hash_code(1, num_codes - 1, hash_array, parents, rank, g_trees);
#endif
        gettimeofday(&mid, NULL);
        cout << "   find edges " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;

    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: " << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec - all_st.tv_usec) / 1e6 << "s using DaC" <<endl;
}
***/
void partition_linear_opt(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<pair<uint, uint>>& edges, float sample_rate)
{
            #ifdef MSTGENERATION
            cout << "ENUM !!!!!!!!!!" << endl;
            #endif
    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    //assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;
    if (M>8) {
        random_shuffle(combinations.begin(), combinations.end());
        combinations.resize(round(combinations.size() * sample_rate));
    }
    vector<pair<uint128_t, uint>> hash_array;
    hash_array.resize(num_codes);
    cout << combinations.size() << " combinations" << endl;
    for (auto k = 0; k < combinations.size(); k++) {
        if (num_codes >= 1000000000) cout << k << " " ;
        //    cout << k << " th combination start " << get_current_time_str() << endl;
        // calculate all hash values
        //vector<pair<uint64_t, uint>> hash_array;
        //cout << "   resize hash array " << get_current_time_str() << endl;
        //cout << "   calculate hash code " << get_current_time_str() << endl;

        gettimeofday(&beg, NULL);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint128_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
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
        for (uint i = 0; i < num_codes; i ++) {
            uint end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            #ifdef MSTGENERATION
            // check each pair
            for (uint j = i; j < end; j ++) {
                for (uint k = j + 1; k < end; k ++) {
                    uint start_i = hash_array[j].second;
                    uint end_i = hash_array[k].second;
                    uint x = find_set(parents, start_i);
                    uint y = find_set(parents,end_i);
                    //if (s_id >= e_id) continue;
                    if (x != y) {
                        // add this edge into the tree
                        if (rank[x] > rank[y]) parents[y] = x;
                        else parents[x] = y;
                        if (rank[x] == rank[y]) rank[y] ++;

                        edges.emplace_back(end_i, start_i);
                        edges.emplace_back(start_i, end_i);
                    }
                }
            }
            #else
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
                    edges.emplace_back(start_i, end_i);
                }
            }
            #endif
            i = end - 1;
        }
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

void partition_linear_opt(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees, float sample_rate)
{
            #ifdef MSTGENERATION
            cout << "ENUM !!!!!!!!!!" << endl;
            #endif
    timeval beg, mid, mid1, end, all_st, all_en; 
    gettimeofday(&all_st, NULL);
    cout << "Find diff = " << DIFF_BY << endl;
    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    //assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;
    if (M>8) {
        random_shuffle(combinations.begin(), combinations.end());
        combinations.resize(round(combinations.size() * sample_rate));
    }
    vector<pair<uint128_t, uint>> hash_array;
    hash_array.resize(num_codes);
    cout << combinations.size() << " combinations" << endl;
    for (auto k = 0; k < combinations.size(); k++) {
        if (num_codes >= 1000000000) cout << k << " " ;
        //    cout << k << " th combination start " << get_current_time_str() << endl;
        // calculate all hash values
        //vector<pair<uint64_t, uint>> hash_array;
        //cout << "   resize hash array " << get_current_time_str() << endl;
        //cout << "   calculate hash code " << get_current_time_str() << endl;

        gettimeofday(&beg, NULL);

        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint128_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint128_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
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
        for (uint i = 0; i < num_codes; i ++) {
            uint end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            #ifdef MSTGENERATION
            // check each pair
            for (uint j = i; j < end; j ++) {
                for (uint k = j + 1; k < end; k ++) {
                    uint start_i = hash_array[j].second;
                    uint end_i = hash_array[k].second;
                    uint x = find_set(parents, start_i);
                    uint y = find_set(parents,end_i);
                    //if (s_id >= e_id) continue;
                    if (x != y) {
                        // add this edge into the tree
                        if (rank[x] > rank[y]) parents[y] = x;
                        else parents[x] = y;
                        if (rank[x] == rank[y]) rank[y] ++;

                        g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                        g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                    }
                }
            }
            #else
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

                    g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                    g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                }
            }
            #endif
            i = end - 1;
        }
        gettimeofday(&mid, NULL);
        //cout << "   find edges " <<mid.tv_sec - beg.tv_sec + (mid.tv_usec - beg.tv_usec) / 1e6 << endl;
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
    gettimeofday(&all_en, NULL);
    cout << "   Find Edge uses: " << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec - all_st.tv_usec) / 1e6 << "sec" <<endl;
}


void partition_linear(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{
    int PART_SIZE = ceil(num_codes / PART_NUM);
    //assert(PART_SIZE * PART_NUM == num_codes);

    cerr << PART_SIZE << " " << num_codes << " " << PART_NUM << " test " << endl;

    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;

    for (auto k = 0; k < combinations.size(); k++) {
        // calculate all hash values
        vector<pair<uint64_t, uint>> hash_array;
        hash_array.resize(num_codes);
        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint64_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
            hash_array[l] = make_pair(hash, l);
        }
        // sort the hash codes
        sort(hash_array.begin(), hash_array.end(), [](const pair<uint64_t, uint32_t>& a, const pair<uint64_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });
        
        // traverse hash array
        for (int i = 0; i < num_codes; i ++) {
            int end = i+1;
            for (; end < num_codes; end ++) {
                if (hash_array[end].first != hash_array[i].first)
                    break;
            }
            // check each pair
            for (int ii = i; ii < end; ii ++) {
                for (int j = ii + 1; j < end; j ++) {
                    
                    uint start_i = hash_array[ii].second;
                    uint end_i = hash_array[j].second;
                    uint x = find_set(parents, start_i);
                    uint y = find_set(parents,end_i);
                    //if (s_id >= e_id) continue;
                    if (x != y) {
                        // add this edge into the tree
                        if (rank[x] > rank[y]) parents[y] = x;
                        else parents[x] = y;
                        if (rank[x] == rank[y]) rank[y] ++;

                        g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                        g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                    }
                }
            }
            i = end - 1;
        }
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}

void partition_new(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{
    int PART_SIZE = ceil(num_codes / PART_NUM);
    //assert(PART_SIZE * PART_NUM == num_codes);

    cerr << PART_SIZE << " " << num_codes << " " << PART_NUM << " test " << endl;

    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;

    for (auto k = 0; k < combinations.size(); k++) {
        // calculate all hash values
        vector<pair<uint64_t, uint>> hash_array;
        hash_array.resize(num_codes);
        #pragma omp parallel for
        for (auto l = 0; l < num_codes; l ++) {
            uint64_t hash = 0x0000000000000000ULL;
            for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
            hash_array[l] = make_pair(hash, l);
        }
        // sort the hash codes
        sort(hash_array.begin(), hash_array.end(), [](const pair<uint64_t, uint32_t>& a, const pair<uint64_t, uint32_t>&b) -> bool {
            return a.first < b.first;
        });

        vector<int> tasks;
        for (int i = 0; i < PART_NUM; i ++) tasks.push_back(i);
        vector<int> marks(PART_NUM);
        for (int i = 0; i < PART_NUM; i ++) marks[i] = 0;
        int batch_num = PART_NUM/NUM_THREAD;

        for (int batch = 0; batch < batch_num; batch ++) {
            vector<vector<pair<uint, uint>>> edge_lists(NUM_THREAD);
            omp_set_dynamic(0);
            omp_set_num_threads(NUM_THREAD);
#pragma omp parallel num_threads(NUM_THREAD)
            {
                int thread_id = omp_get_thread_num();
                if (batch * NUM_THREAD + thread_id < tasks.size()) {
                    int j = tasks[batch * NUM_THREAD + thread_id];
                    marks[j] = 1;
                    for (auto l = j * PART_SIZE; l < (j+1)*PART_SIZE && l < num_codes; l++) {
                        uint64_t hash = 0x0000000000000000ULL;
                        for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                            hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
                        // binary search here
                        uint index = binary_search(hash_array, hash, 0, 
                                                    num_codes-1);
                        if (index != -1) {
                            // enumerate edges
                            // search for previous hash codes
                            if (index != 0) {
                                int iter = index-1;
                                while (iter >= 0 && hash_array[iter].first == hash) {
                                    // this is a pair
                                    uint s_id = hash_array[iter].second;
                                    uint e_id = hash_array[index].second;
                                    iter --;
                                    //if (s_id >= e_id) continue;
                                    if (find_set_read_only(parents, s_id) != 
                                            find_set_read_only(parents, e_id)) {
                                        edge_lists[thread_id].emplace_back(s_id, e_id);
                                    }
                                }
                                iter = index + 1;
                                while (iter < num_codes && hash_array[iter].first == hash) {
                                    // this is a pair
                                    uint s_id = hash_array[iter].second;
                                    uint e_id = hash_array[index].second;
                                    iter ++;
                                    //if (s_id >= e_id) continue;
                                    if (find_set_read_only(parents, s_id) != 
                                            find_set_read_only(parents, e_id)) {
                                        edge_lists[thread_id].emplace_back(s_id, e_id);
                                    }
                                }

                            }
                        }
                    }
                }
            }
            long long num_edges=0;
            for (auto &list : edge_lists) {
                num_edges += list.size();
                for (auto &pair: list) {
                    uint start_i = pair.first;
                    uint end_i = pair.second;
                    int x = find_set(parents, start_i);
                    int y = find_set(parents,end_i);
                    if (x == y) continue;   // already in the same set
                    if (rank[x] > rank[y]) parents[y] = x;
                    else parents[x] = y;
                    if (rank[x] == rank[y]) rank[y] ++;

                    g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                    g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));

                }
            }

        }
        for (int i = 0; i < PART_NUM; i ++) 
            if (marks[i] == 0) 
                cout << "task " << i << " is not done" << endl;
    }

}

uint binary_search(const vector<pair<uint64_t, uint32_t>>& hash_array, 
                   uint64_t hashcode, uint start, uint end) {
    if (start == end) return -1;
    uint mid = (start + end)/2;
    uint64_t left_hash = hash_array[start].first;
    if (hashcode == left_hash) return start;
    uint64_t right_hash= hash_array[end].first;
    if (hashcode == right_hash) return end;
    uint64_t mid_hash  = hash_array[mid].first;
    if (hashcode == mid_hash) return mid;

    if (hashcode < mid_hash) return binary_search(hash_array, hashcode, start, mid);
    return binary_search(hash_array, hashcode, mid, end);
}
void partition(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{
    int PART_SIZE = ceil(num_codes / PART_NUM);
    //assert(PART_SIZE * PART_NUM == num_codes);

    cerr << PART_SIZE << " " << num_codes << " " << PART_NUM << " test " << endl;

    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;

    int num_of_combs = combinations.size();
    vector<tuple<int, int, int>> tasks;
    for (auto i = 0; i < PART_NUM; i++)
        for (auto j = i; j < PART_NUM; j++)
            for (auto k = 0; k < num_of_combs; k++)
                tasks.emplace_back(i, j, k);

    int num_of_tasks = tasks.size();
    int batch_num = num_of_tasks / NUM_THREAD;

    for (int batch = 0; batch < batch_num; batch++)
    {
        vector<vector<pair<uint, uint>>> edge_lists(NUM_THREAD);
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(NUM_THREAD); 
#pragma omp parallel num_threads(NUM_THREAD)
        {
            int thread_id = omp_get_thread_num();
            //cout << " PART_NUM " << PART_NUM << endl;
            //cout << " com_NUm " << num_of_combs << endl;
            //cout << " task_num " << tasks.size() << endl;
            //cout << " thread_id " << thread_id << endl;
            if (batch * NUM_THREAD + thread_id < tasks.size())
            {
                int i = get<0>(tasks[batch * NUM_THREAD + thread_id]);
                int j = get<1>(tasks[batch * NUM_THREAD + thread_id]);
                int k = get<2>(tasks[batch * NUM_THREAD + thread_id]);
                //cout << " i " << i << endl;
                //cout << " j " << j << endl;
                //cout << " k " << k << endl;

                unordered_map<uint64_t, vector<int>> invindex;
                for (auto l = i * PART_SIZE; l < (i + 1) * PART_SIZE && l < num_codes; l++)
                {
                    // if (l % 10000000 == 0) {
                    //cout <<"    " << k << " First loop " <<l <<" " <<get_current_time_str() << endl;
                    // }
                    uint64_t hash = 0x0000000000000000ULL;
                    for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                        hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));

                    invindex[hash].emplace_back(l);
                }
                //cout << "invindex size: " << invindex.size() << endl;

                // lookup inverted index to get code
                for (auto l = j * PART_SIZE; l < (j + 1) * PART_SIZE && l < num_codes; l++)
                {
                    // if (l % 10000000 == 0 && k % 8 == 0) {
                    //    cout <<"    iter = "<<iter<<" k = " << k << " second loop = " <<l <<" " <<get_current_time_str() << endl;
                    // }
                    uint64_t hash = 0x0000000000000000ULL;
                    for (auto it = combinations[k].begin(); it != combinations[k].end(); it++)
                        hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));

                    if (invindex.count(hash))
                    {
                        for (auto vit = invindex[hash].begin(); vit != invindex[hash].end(); vit++)
                        {
                            if (*vit < l) {

                                int start_i = *vit, end_i = l;
                                if (find_set_read_only(parents, start_i) != 
                                        find_set_read_only(parents, end_i)) {
                                    edge_lists[thread_id].emplace_back(start_i, end_i);
                                }
                            }
                        }
                    }
                }
            }
        }
        long long num_edges=0;
        for (auto &list : edge_lists) {
            num_edges += list.size();
            for (auto &pair: list) {
                uint start_i = pair.first;
                uint end_i = pair.second;
                int x = find_set(parents, start_i);
                int y = find_set(parents,end_i);
                if (x == y) continue;   // already in the same set
                if (rank[x] > rank[y]) parents[y] = x;
                else parents[x] = y;
                if (rank[x] == rank[y]) rank[y] ++;

                g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));

            }
        }
        if (batch % 1000 == 0) {
            cout << batch << " batches have been processed " << 
                get_current_time_str() << endl;
            double vm, rss;
            process_mem_usage(vm, rss);
            cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
            cout << "Total number of pairs: " << num_edges << endl << endl;
        }
    }
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}

void partition_wang(unsigned char* codes, int M, int K, int LOG_K, int num_codes, 
            int PART_NUM, int DIFF_BY, uint* parents, uint* rank, 
            atomic_flag& lock, vector<vector<pair<int, float>>>& g_trees)
{
    int PART_SIZE = ceil(num_codes / PART_NUM);
    //assert(PART_SIZE * PART_NUM == num_codes);

    cerr << PART_SIZE << " " << num_codes << " " << PART_NUM << " test " << endl;

    vector<vector<int>> combinations;

    assert(M > DIFF_BY);
    nchoosek(M, M - DIFF_BY, combinations);

    assert(M * LOG_K <= 64);
    atomic<long long> count(0);
    // enumerate every data partition pairs;
    cout << "For loop begins " << get_current_time_str() << endl;
    
    for (auto k = 0; k < combinations.size(); k++) {
        vector<unordered_map<uint64_t, vector<int>>> invindices;
        invindices.resize(PART_NUM);
        // build inverted list in parallel
        #pragma omp parallel for if (parallelism_enabled)
        for (auto i = 0; i < PART_NUM; i++) {
            for (auto l = i * PART_SIZE; l<(i+1)*PART_SIZE && l<num_codes; l++) {
                uint64_t hash = 0x0000000000000000ULL;
                for (auto it = combinations[k].begin(); 
                                    it != combinations[k].end(); it++) {
                    hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
                }
                invindices[i][hash].emplace_back(l);
            }
        }
        // enum all <i,j> pairs and prepare edge lists for <i,j> pairs
        vector<pair<int, int>> ij_pairs;
        vector<vector<pair<uint, uint>>> edge_lists(PART_NUM*PART_NUM);
        
        for (int i = 0; i < PART_NUM; i ++) {
            for (int j = i; j < PART_NUM; j ++) {
                ij_pairs.emplace_back(i,j);
                int index = i * PART_NUM + j;
            }
        }
        
        // process each <i,j> pair in parallel to find edges

        #pragma omp parallel for if (parallelism_enabled)
        for (int idx = 0; idx < ij_pairs.size(); idx ++) {
            int i = ij_pairs[idx].first;
            int j = ij_pairs[idx].second;
            long long edge_list_index = i*PART_NUM+j;
            for (uint l = j*PART_SIZE; l < (j+1)*PART_SIZE && l < num_codes; l ++){
                uint64_t hash = 0x0000000000000000ULL;
                for (auto it = combinations[k].begin(); 
                          it!= combinations[k].end(); it++) {
                    hash |= (static_cast<uint64_t>(codes[GetCodeIndex(M, l, *it)]) << (LOG_K * (*it)));
                }
                if (invindices[i].count(hash)) {
                    for (auto vit = invindices[i][hash].begin(); 
                                vit != invindices[i][hash].end(); vit++) {
                        if (*vit < l) {
                        // check union-find set
                            int start_i = *vit, end_i = l;
                            if (find_set_read_only(parents, start_i) != 
                                find_set_read_only(parents, end_i)) {
                                edge_lists[edge_list_index].emplace_back(start_i, end_i);
                            }
                        }
                    }
                }
            } 
        }
        //traverse edge list and perform union-find
        for (auto &list : edge_lists) {
            for (auto pair: list) {
                uint start_i = pair.first;
                uint end_i = pair.second;
                int x = find_set(parents, start_i);
                int y = find_set(parents,end_i);
                if (x == y) continue;   // already in the same set
                if (rank[x] > rank[y]) parents[y] = x;
                else parents[x] = y;
                if (rank[x] == rank[y]) rank[y] ++;

                g_trees[start_i].push_back(make_pair(end_i,DIFF_BY));
                g_trees[end_i].push_back(make_pair(start_i,DIFF_BY));
                
            }
        }
    }
    cout << "Total number of pairs: " << count << endl;
    double vm, rss;
    process_mem_usage(vm, rss);
    cout << "find edge VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
}
/*
int main(int argc, char ** argv) 
{
  srand (time(NULL));
  int ncodes = atoi(argv[1]);
  // read uchar
  codes = new uchar[ncodes * M];
  for (int i = 0; i < ncodes * M; i++)
    codes[i] = rand() % 10;

  num_codes = ncodes;

  for (int i = 0; i < ncodes; i++)
    print_code(i);

  cout << "finished printing " << endl;

  partition(atoi(argv[2]), atoi(argv[3]));
  delete[] codes;
}
*/
