#include "pqbase.h"

bool PQBase::cmp(const pair<int, double> &t1, const pair<int, double> &t2) {
    return t1.second > t2.second;
}
void PQBase::create_dist_tables() {
    dist_tables = vector<vector<float> > (PQ_M, vector<float>(PQ_K*PQ_K,0));
    for (int i = 0; i < PQ_M; i ++) {
        for (int j = 0; j < PQ_K; j ++) {
            for (int k = 0; k < PQ_K; k++) {
                float dist = 0;
                for (int idx = 0; idx < m_Ds; idx ++) {
                    dist+=pow(m_codewords[i][j][idx]-m_codewords[i][k][idx],2);
                }
                dist_tables[i][j*PQ_K+k] = dist;
            }
        }
    }
}

float PQBase::cal_distance_by_tables(int a, int b) {
    float sum = 0;
    for (int m = 0; m < PQ_M; m ++) {
        int c_a = (int) vecs[(long long)a*PQ_M+m];
        int c_b = (int) vecs[(long long)b*PQ_M+m];
        sum += dist_tables[m][c_a*m_Ks+c_b]; // m_Ks = PQ_K
    }
    return sum;
}

void PQBase::create_MKKTable() {
    dist_tables = vector<vector<float> > (PQ_M, vector<float>(PQ_K*PQ_K,0));
    for (int i = 0; i < PQ_M; i ++) {
        for (int j = 0; j < PQ_K; j ++) {
            for (int k = 0; k < PQ_K; k++) {
                float dist = 0;
                for (int idx = 0; idx < m_Ds; idx ++) {
                    dist+=pow(m_codewords[i][j][idx]-m_codewords[i][k][idx],2);
                }
                dist_tables[i][j*PQ_K+k] = dist;
            }
        }
    }
    // checked
}
float PQBase::cal_dist_bound(Node* node) {
    if (node->n_children == 0) return node->distance;
    node->max_dist = 0;
    // traverse each child
    for (int i = 0; i < node->n_children; i ++) {
        float dist_i = 0;
        for (tuple<uchar, uchar, uchar> tp : node->children[i]->transform) {
            int m = (int)std::get<0>(tp);
            int a = (int)std::get<1>(tp);
            int b = (int)std::get<2>(tp);
            dist_i += dist_tables[m][a*PQ_K+b];
        }
        dist_i += cal_dist_bound(node->children[i]);
        if (dist_i > node->max_dist) node->max_dist = dist_i;
    }
    return node->max_dist;
}
float PQBase::cal_dist_from_query(int id) {
    float sum = 0;
    for (int i = 0; i < PQ_M; i++) {
        int idx = (int)vecs[id*PQ_M+i];
        sum += m_sub_distances[i][idx];
    }
    return sum;
}
// find the optimum root for each tree in one pass
void PQBase::adjust_trees() {
    // output: change the parent of each tree to the a propriate node
    tree_mark.resize(n); // because there are two directed edges for each pair
    fill(tree_mark.begin(), tree_mark.end(), 0);

    unordered_map<long long, vector<uint>> trees;
    for (uint i = 0 ; i < N; i ++) {
        uint p = parents[i];
        if (trees.find(p) == trees.end()) {
            trees.emplace(p, vector<uint>(1, i));
        } else {
            trees[p].push_back(i);
        }
    }

    n_trees = trees.size();
    
    uint max_size = 0;
    // find the largest tree
    for (auto iter : trees) {
        if (iter.second.size() > max_size)
            max_size = iter.second.size();
    }
    // prepare BFS queue
    uint* bfs_queue = new uint[max_size];
    uint bfs_index = 0, queue_size = 0;
    uint* parent_pos= new uint[max_size];
    uint* highest_child=new uint[max_size];
    uint* heights   = new uint[max_size];
    memset(parent_pos, 0, sizeof(uint)*max_size);
    memset(highest_child, 0, sizeof(uint)*max_size);
    memset(heights, 0, sizeof(uint)*max_size);
    for (auto iter : trees) {
        // get nodes and root in the tree
        vector<uint>& nodes = iter.second;
        long long root_id = iter.first;
        int min_height = -1;
        // reset tree mark
        for (uint node : nodes) tree_mark[node] = 0;
        // reset bfs queue
        bfs_index = queue_size = 0;
        // start bfs
        bfs_queue[queue_size++] = root_id;
        tree_mark[root_id] = 1;
        while (bfs_index != queue_size) {
            uint node = bfs_queue[bfs_index];
            for (uint j = 0; j < get_degree(node); j ++) {
                uint child = get_neighbor(node, j);
                if (child >= n) {
                    cout << child << endl;
                    cout << j << endl;
                    cout << node << endl;
                }
                if (tree_mark[child] == 0) {
                    tree_mark[child] = 1;
                    bfs_queue[queue_size] = child;
                    parent_pos[queue_size] = bfs_index;
                    queue_size++;
                }
            }
            bfs_index ++;
        }
        //memset(heights, 0, sizeof(uint)*max_size);
        // set heights, start from the end
        for (int pos = bfs_index-1; pos > 0; pos--) {
            uint parent_p = parent_pos[pos];
            if (heights[parent_p] < heights[pos] + 1) {
                heights[parent_p] = heights[pos] + 1;
                highest_child[parent_p] = pos;
            }
        }
        min_height = heights[0];
        // find the highest 2 children of the root
        int max_child_height = 0;
        int second_child_height=0;
        for (uint i = 1; i < bfs_index; i ++) {
            if (parent_pos[i] != 0) break;
            int height = heights[i];
            if (height >= max_child_height) {
                // put current max to the second
                second_child_height = max_child_height;
                // update current max info
                max_child_height = heights[i];
            } else if (height >= second_child_height) {
                second_child_height = height;
            }
        }
        if (max_child_height == 0) continue;    // there is only 1 node in tree
        // now we need to find a propriate root
        int lowest_height = (max_child_height + second_child_height + 1)/2+1;
        int deviation = heights[0]-lowest_height;
        uint root_pos = 0;

        for (int i = 0; i < deviation; i ++) {
            // go to its highest child
            root_pos = highest_child[root_pos];
        }
        root_id = bfs_queue[root_pos];
        // update root info on union-find set
        for (uint node : nodes) parents[node] = root_id;
        // reset heights here
        for (uint i = 0; i < bfs_index; i ++) heights[i] = 0;
    }
}
// decrease height of a tree by 1 each time
void PQBase::rotate_trees() {
    // output: change the parent of each tree to the a propriate node
    tree_mark.resize(n); // because there are two directed edges for each pair
    fill(tree_mark.begin(), tree_mark.end(), 0);

    unordered_map<long long, vector<uint>> trees;
    for (uint i = 0 ; i < N; i ++) {
        uint p = parents[i];
        if (trees.find(p) == trees.end()) {
            trees.emplace(p, vector<uint>(1, i));
        } else {
            trees[p].push_back(i);
        }
    }

    n_trees = trees.size();
    
    uint max_size = 0;
    // find the largest tree
    for (auto iter : trees) {
        if (iter.second.size() > max_size)
            max_size = iter.second.size();
    }
    // prepare BFS queue
    uint* bfs_queue = new uint[max_size];
    uint bfs_index = 0, queue_size = 0;
    uint* parent_pos= new uint[max_size];
    uint* heights   = new uint[max_size];
    for (auto iter : trees) {
        // get nodes and root in the tree
        vector<uint>& nodes = iter.second;
        // DBEUG
        //if (nodes.size() > 5000) { 
        //    cout << "----------------" << endl;
        //    cout << "tree size is " << nodes.size() << endl;
        //}   // END OF DEBUG
        long long root_id = iter.first;
        uint min_height = (uint)-1;
        bool terminate = false;
        while (terminate == false) {
            // reset tree mark
            for (uint node : nodes) tree_mark[node] = 0;
            // reset bfs queue
            bfs_index = queue_size = 0;
            // start bfs
            bfs_queue[queue_size++] = root_id;
            tree_mark[root_id] = 1;
            while (bfs_index != queue_size) {
                uint node = bfs_queue[bfs_index];
                for (uint j = 0; j < g_trees[node].size(); j ++) {
                    uint child = g_trees[node][j].first;
                    if (tree_mark[child] == 0) {
                        tree_mark[child] = 1;
                        bfs_queue[queue_size] = child;
                        parent_pos[queue_size] = bfs_index;
                        queue_size++;
                    }
                }
                bfs_index ++;
            }
            memset(heights, 0, sizeof(uint)*max_size);
            // set heights, start from the end
            for (uint pos = bfs_index-1; pos > 0; pos--) {
                uint parent_p = parent_pos[pos];
                if (heights[parent_p] < heights[pos] + 1) {
                    heights[parent_p] = heights[pos] + 1;
                }
            }
            // DEBUG
            //if (nodes.size() > 5000 ) {
            //    cout << "Root height " << heights[0] << endl;
            //    cout << "Min height " << min_height << endl;
            //    cout << "bfs_index " << bfs_index << endl;
            //    //for (uint i = 0; i < bfs_index; i ++) {
            //    //    cout << parent_pos[i] << " height = " << heights[i]<< endl;
            //    //}
            //}
            // END of DEBUG
            if (min_height <= heights[0]) {
                // this rotation worked
                terminate = true;
            } else {
                min_height = heights[0];
                // DEBUG
                //if (nodes.size() > 5000) { 
                //    cout << "   -----------------" << endl;
                //    cout << "previous height " << min_height << endl;
                //}   // DEBUG
                // check each child of root
                uint max_child_height = 0;
                uint new_root_id = root_id;
                for (uint i = 1; i < bfs_index; i ++) {
                    if (parent_pos[i] != 0) break;
                    if (heights[i] > max_child_height) {
                        max_child_height = heights[i];
                        new_root_id = bfs_queue[i];
                    }
                }
                // set root id for next round
                // DEBUG
                //if (nodes.size() > 5000) { 
                //    cout << "Max child height is  " <<  max_child_height << endl;
                //}   // DEBUG
                root_id = new_root_id;
            }
        }
        // update root info on union-find set
        for (uint node : nodes) parents[node] = root_id;
    }
}
inline uint PQBase::get_degree(uint i) {
    return offsets[i+1] - offsets[i];
}
inline uint PQBase::get_neighbor(uint i, uint j) {
    if (i == 9984) cout << "get_neighbor base " << offsets[i] << endl;
    return sparse_row[offsets[i]+j];
}
void PQBase::write_groundtruth(std::string file_path, int M, int K, vector<vector<pair<uint, float>>>& results, int n_queries, int top_k)
{
    std::ofstream ofs(file_path);
    cout << file_path << endl;
    assert(ofs.is_open());

    ofs << n_queries << "," << top_k << "\n";

    for(int i = 0; i < n_queries; i ++) {
        //for (int tk = top_k-1; tk >= 0; tk --) {
        sort(results[i].begin(), results[i].end(), [](const pair<uint,float>& a, const pair<uint, float>& b) {
            return a.second < b.second;
        });
        for (int tk = 0; tk < top_k; tk ++) {
            ofs << results[i][tk].first << ","<<results[i][tk].second<<",";
        }
        ofs << "\n";
    }
}
void PQBase::read_groundtruth(std::string file_path, int M, int K, vector<vector<pair<uint, float>>>& results, int n_queries, int top_k)
{
    std::ifstream ifs(file_path);
    cout << file_path << endl;
    assert(ifs.is_open());
    char c1,c2;

    cout << M << " " << K << endl;
    ifs >> n_queries >> c1 >> top_k;
    cout << n_queries << " " << top_k << endl;
    results.resize(n_queries);

    for(int i = 0; i < n_queries; i ++) {
        results[i].resize(top_k);
        for (int tk = 0; tk < top_k; tk ++) {
            ifs >> results[i][tk].first >> c1 >> results[i][tk].second >> c1;
            if (i == 0 && tk < 10) cout << "GT: " << results[i][tk].first << " " << results[i][tk].second << endl;
        }
    }
}
