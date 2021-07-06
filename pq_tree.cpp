#include "pq_tree.h"
#include <queue>
static string get_current_time_str() {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
    std::string str(buffer);

    return str;
}
float CalSquaredDistance(vector<float> a, vector<float> b) {
    int length = a.size();
    if (length != b.size()) {
        cout << "lengths of the two input vectors are not equal" << endl;
    }
    float sum = 0;
    for (int i = 0; i < length; i ++) {
        sum += pow(a[i]-b[i], 2);
    }
    return sum;
}

bool pair_sorter(pair<float, vector<float>> a, 
        pair<float, vector<float>> b) {
    return a.first < b.first;
}

bool bound_sorter(pair<int, float> a, pair<int, float> b) {
    return a.second < b.second;
}

bool bitset_sorter(bitset<CL> a, bitset<CL> b) {
    return a.to_ulong() < b.to_ulong();
}

PQTree::PQTree(vector<PQ::Array> codewords) {
    m_codewords = codewords;
    m_codewords_t = codewords;
    m_M = m_codewords_t.size();
    m_Ks = m_codewords_t[0].size();
    m_Ds = m_codewords_t[0][0].size();
}

PQTree::PQTree(vector<PQ::Array> codewords, BitVecs* bitvecs) {
    m_codewords = codewords;
    m_codewords_t = codewords;
    m_M = m_codewords_t.size();
    m_Ks = m_codewords_t[0].size();
    m_Ds = m_codewords_t[0][0].size();
    database = bitvecs;
    //m_sub_distances.reserve(m_M); // M x K
    m_sub_distances = new float*[m_M];
    m_sub_bounds = new float*[m_M];
    for (int i = 0; i < m_M; i++) {
        m_sub_distances[i] = new float[m_Ks];
        m_sub_bounds[i] = new float[m_Ks+m_Ks];
        memset(m_sub_distances[i], 0, sizeof(float)*m_Ks);
        memset(m_sub_bounds[i],    0, sizeof(float)*(m_Ks + m_Ks));
    }
    cout << "m_M: " << m_M << " m_Ks: " << m_Ks << " m_Ds: " << m_Ds << endl;
}
#ifdef PLAIN_PQ
PQTree::PQTree(vector<PQ::Array> codewords, uchar* vecs, long length) {
    m_codewords = codewords;
    m_codewords_t = codewords;
    m_M = m_codewords_t.size();
    m_Ks = m_codewords_t[0].size();
    m_Ds = m_codewords_t[0][0].size();
    plain_database = vecs;
    N = length;
    m_sub_distances = new float*[m_M];
    for (int i = 0; i < m_M; i++) {
        m_sub_distances[i] = new float[m_Ks];
        memset(m_sub_distances[i], 0, sizeof(float)*m_Ks);
    }
    cout << "m_M: " << m_M << " m_Ks: " << m_Ks << " m_Ds: " << m_Ds << endl;
    dist_hist.resize(m_M+1);
    fill(dist_hist.begin(), dist_hist.end(), 0 );
}
#endif
void PQTree::DichotomizeCodewords(int start, int end) {
    int M = m_codewords_t.size();
    int Ks = m_codewords_t[0].size();
    // Do learn
    for(int m = 0; m < M; ++m){ // for each subspace
        //cout << "learning m: " << m << " / " << M << endl;

        // Focus on sub space
        PQ::Array& sub_words = m_codewords_t[m];
        // cluster a subset of the k centroids
        PQ::Array sub_node = PQ::Array(sub_words.begin()+start,
                                            sub_words.begin()+end);
        cv::Mat svec = PQ::ArrayToMat(sub_node);

        // Do k-means with k=2
        cv::Mat label;
        cv::Mat center;
        cv::kmeans(svec, 2, label,
                   cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 1000, 1),
                   3, cv::KMEANS_PP_CENTERS, center);

        // get two centers for each partition
        PQ::Array centers = PQ::MatToArray(center);
        // calculate benefit for each centroids within the range
        vector<pair<float, vector<float>>> dist_pairs;
        for (int i = start; i < end; i ++) {
            vector<float> centroid = sub_node[i-start];
            float dist0 = CalSquaredDistance(centroid, centers[0]);
            float dist1 = CalSquaredDistance(centroid, centers[1]);
            dist_pairs.push_back(make_pair(dist0-dist1, centroid));
        }
        // sort the pairs
        sort(dist_pairs.begin(), dist_pairs.end(), pair_sorter);
        // relocate centroids
        for (int i = start; i < end; i ++) {
            m_codewords_t[m][i] = dist_pairs[i-start].second;
        }
    }
    if (end - start < 4) return;
    // recursively dichotomize the centroids
    DichotomizeCodewords(start, (start+end)/2);
    DichotomizeCodewords((start+end)/2, end);
}

vector<PQ::Array> PQTree::GetCodewordsT() {
    return m_codewords_t;
}

bitset<CL> PQTree::Encode(const vector<float> &vec) const
{
    assert((int) vec.size() == m_Ds * m_M);
    bitset<CL> bitcode;
    vector<uchar> code(m_M);
#ifdef ENCODE_DEBUG
    // DEBUG
    for (int i = 0; i < m_M; i++) {
        int index = (int)code[i];
        for (int j = 0; j < m_Ds; j ++) {
            cout << vec[i*m_Ds + j] << " "; // DEBUG
        }
        cout << endl << endl;
    }
#endif
    for(int m = 0; m < m_M; ++m){
        float min_dist = FLT_MAX;
        int min_ks = -1;

        // find the nearest codeword
        for(int ks = 0; ks < m_Ks; ++ks){
            float dist = 0;
            for(int ds = 0; ds < m_Ds; ++ds){
                float diff = vec[m * m_Ds + ds] - m_codewords_t[m][ks][ds];
                dist += diff * diff;
            }
            if(dist < min_dist){
                min_dist = dist;
                min_ks = ks;
            }
        }
        assert(min_ks != -1);
        code[m] = (uchar) min_ks;
    }
    for (int i = 0; i < m_M; i ++) {
        int sub_code= (int)code[i];
        for (int j = 0; j < CL/m_M; j ++) {
            int idx = (m_M-1-i)+j*m_M;
            bitcode.set(idx, sub_code & 1);
            sub_code >>= 1;
        }
    }
#ifdef ENCODE_DEBUG
    // DEBUG
    for (int i = 0; i < m_M; i++) {
        int index = (int)code[i];
        for (int j = 0; j < m_Ds; j ++) {
            cout << m_codewords[i][index][j] << " "; // DEBUG
        }
        cout << endl << endl;
    }
    cout << bitcode << endl;
    exit(0);
#endif

    return bitcode;
}

vector<uchar> PQTree::EncodePlain(vector<float> &vec)
{
    int D = m_Ds * m_M;
    int vec_d = vec.size();
    for (int i = vec_d; i < D; i ++) {
        vec.emplace_back(0);
    }
    assert((int) vec.size() == m_Ds * m_M);
    vector<uchar> code;
    if (m_Ks > 256)
        code.resize(m_M*2);
    else
        code.resize(m_M);
#ifdef ENCODE_DEBUG
    // DEBUG
    for (int i = 0; i < m_M; i++) {
        int index = (int)code[i];
        for (int j = 0; j < m_Ds; j ++) {
            cout << vec[i*m_Ds + j] << " "; // DEBUG
        }
        cout << endl << endl;
    }
#endif
    for(int m = 0; m < m_M; ++m){
        float min_dist = FLT_MAX;
        int min_ks = -1;

        // find the nearest codeword
        for(int ks = 0; ks < m_Ks; ++ks){
            float dist = 0;
            for(int ds = 0; ds < m_Ds; ++ds){
                float diff = vec[m * m_Ds + ds] - m_codewords[m][ks][ds];
                dist += diff * diff;
            }
            if(dist < min_dist){
                min_dist = dist;
                min_ks = ks;
            }
        }
        assert(min_ks != -1);
        if (m_Ks > 256) {
            code[m*2] = min_ks/256;
            code[m*2+1] = min_ks%256;
        } else
            code[m] = (uchar) min_ks;
    }
#ifdef ENCODE_DEBUG
    // DEBUG
    for (int i = 0; i < m_M; i++) {
        int index = (int)code[i];
        for (int j = 0; j < m_Ds; j ++) {
            cout << m_codewords[i][index][j] << " "; // DEBUG
        }
        cout << endl << endl;
    }
    for (int i = 0; i < m_M; i++) {
        cout << bitset<8>((int)code[i]) << endl;
    }
    exit(0);
#endif
    return code;
}

BitVecs PQTree::Encode(const vector<vector<float> > &vecs) const
{
    int M = m_codewords_t.size();
    BitVecs codes((int) vecs.size(), M);
    for(int i = 0; i < (int) vecs.size(); ++i){
        codes.SetVec(i, Encode(vecs[i]));
    }
    return codes;
}

uchar* PQTree::EncodePlain(vector<vector<float> > &vecs)
{
    int M = m_codewords_t.size();
    uchar* codes = new uchar[vecs.size()*m_M];
    memset(codes, 0, sizeof(uchar)*vecs.size()*m_M);
    for(int i = 0; i < (int) vecs.size(); ++i){
        vector<uchar> one_code = EncodePlain(vecs[i]);
        for (int j = 0; j < 4; j ++) {
            codes[i*m_M+ j] = one_code[j];
        }
    }
    return codes;
}
vector<pair<int, float> > PQTree::SampledQuery(vector<float> &query, int id) {
    //cout << get_current_time_str() << endl;
    vector<pair<int, float>> results(1);
    // calculate sub-distances
    #ifdef QUERY_DEBUG
    cout << "Encode query vector : " << endl;
    EncodePlain(query);
    cout << "End of query vector encoding" << endl;
    #endif
    for (int i = 0; i < m_M; i++) {
        for (int j = 0; j < m_Ks; j ++) {
            float dist_2 = 0;
            for (int k = 0; k < m_Ds; k ++) {
                float dist = m_codewords[i][j][k] - query[i*m_Ds+k];
                dist_2 += dist * dist;
            }
            m_sub_distances[i][j] = dist_2;
        }
    }

    // find the nearest centroids (with centroid id and distance) DEBUG 
    // TODO delete this chunk
    float debug = 0; // DEBUG
    float debug_max = 0;
    vector<pair<int, float>> closest_centroids(m_M);
    for (int i = 0 ; i < m_M; i ++) {
        float min_dist = numeric_limits<float>::max();
        float max_dist = 0;
        int min_id = 0;
        for (int j = 0; j < m_Ks; j ++) {
            if (m_sub_distances[i][j] < min_dist) {
                min_dist = m_sub_distances[i][j];
                min_id = j;
            }
            if (m_sub_distances[i][j] > max_dist) {
                max_dist = m_sub_distances[i][j];
            }
        }
        #ifdef QUERY_DEBUG
        debug+=min_dist; 
        //cout << bitset<8>(min_id) <<" ";// DBEUG
        cout << min_id <<" ";// DBEUG
        debug_max+=max_dist;
        for (int j = 0; j < m_Ds; j++) {
            //cout << m_codewords[i][min_id][j] << " ";// DEBUG
        }
        cout << endl;// DBEUG
        #endif
        closest_centroids[i] = make_pair(min_id, min_dist);
    }
    #ifdef QUERY_DEBUG
    cout << debug << endl; // DEBUG
    cout << debug_max << endl;
    #endif
    float min_dist = FLT_MAX;
    long min_id = -1;
    float max_dist = 0;
    // scan the entire database
    for (long i = 0; i < N; i ++) {
        if (i == id) continue;
        #ifdef QUERY_DEBUG
        if (i == 11114457) {
            cout << "check i =" << i << endl;

            for (int j = 0; j < m_M; j ++) {
                int index = (int)plain_database[i*m_M+j];
                cout << bitset<8>(index) << endl;
            }
            for (int j = 0; j < m_M; j ++) {
                int index = (int)plain_database[i*m_M+j];
                for (int k = 0; k < m_Ds; k ++) {
                    cout << m_codewords[j][index][k] << " ";
                }
                cout << endl << endl;
            }
            exit(0);
        }
        #endif

        float sum = 0;
        for (int j = 0; j < m_M; j ++) {
            int index = (int)plain_database[i*m_M+j];
            sum += m_sub_distances[j][index];
        }
        if (sum < min_dist) {
            min_dist = sum;
            min_id = i;
        }
        // DEBUG
        if (max_dist < sum) max_dist = sum;
    }
        #ifdef QUERY_DEBUG
    cout << "Min dist " << min_dist << " " << min_id << endl;
    cout << "Max dist " << max_dist << endl;
        #endif
    results[0] = make_pair(min_id, min_dist);
    //for (int i = 0; i < m_M; i++) {
    //    int index = (int)plain_database[min_id*m_M+i];
    //    for (int j = 0; j < m_Ds; j ++) {
    //        cout << m_codewords[i][index][j] << " "; // DEBUG
    //    }
    //    cout << endl << endl;
    //}
    //cout << endl;// DBEUG
    //cout << get_current_time_str() << endl;
    //exit(0);
    // check difference
    int dist_m = 0;
    vector<uchar> query_code = EncodePlain(query);
    for (int i = 0; i <m_M; i ++) {
        if (plain_database[min_id*m_M+i] != query_code[i]) dist_m ++;
    }
    dist_hist[dist_m] ++;
    return results;
}
#ifdef PLAIN_PQ
auto comp_max_heap = [](const pair<int, float> &a, const pair<int, float> &b) {
    return a.second < b.second;
};
void print_heap(vector<pair<int, float>>& test) {
    for (int i = 0; i < test.size(); i ++) {
        cout << test[i].first << ", " << test[i].second << "    ";
    }
    cout << endl;
}

void PQTree::QueryPlain(const vector<float> &query, int top_k, vector<pair<uint, float>>& dist_pairs) {
    
    // calculate sub-distances
    for (int i = 0; i < m_M; i++) {
        for (int j = 0; j < m_Ks; j ++) {
            float dist_2 = 0;
            for (int k = 0; k < m_Ds; k ++) {
                float dist = m_codewords[i][j][k] - query[i*m_Ds+k];
                dist_2 += dist * dist;
            }
            m_sub_distances[i][j] = dist_2;
        }
    }

    // scan the entire database
    for (long i = 0; i < N; i ++) {
        
        float sum = 0;
        for (int j = 0; j < m_M; j ++) {
            int index;
            if (PQ_K > 256) 
                index = (int)*((uint16_t*)(plain_database+(i*m_M+j)*2));
            else 
                index = (int)plain_database[i*m_M+j];
            sum += m_sub_distances[j][index];
        }
        dist_pairs[i] = make_pair(i, sum);
    }
}

vector<pair<int, float> > PQTree::QueryPlain(const vector<float> &query, int top_k) {
    //cout << get_current_time_str() << endl;
    vector<pair<int, float>> results(top_k);
    auto cmp_max = [](pair<float, uint>& left, pair<float, uint>& right) {
        return left.first < right.first;
    };
    priority_queue<pair<float, uint>, vector<pair<float, uint>>,
                    decltype(cmp_max)> max_heap(cmp_max);
    //print_heap(results);
    for (int i = 0; i < top_k; i ++) results[i].second = FLT_MAX;
    // calculate sub-distances
    for (int i = 0; i < m_M; i++) {
        for (int j = 0; j < m_Ks; j ++) {
            float dist_2 = 0;
            for (int k = 0; k < m_Ds; k ++) {
                float dist = m_codewords[i][j][k] - query[i*m_Ds+k];
                dist_2 += dist * dist;
            }
            m_sub_distances[i][j] = dist_2;
        }
    }

    // scan the entire database
    for (long i = 0; i < N; i ++) {
        //long start_i = i*m_M;
        uchar* vec = plain_database+i*m_M;
        float sum = 0;
        for (int j = 0; j < m_M; j ++) {
            int index;
            if (PQ_K > 256) 
                index = (int)*((uint16_t*)(plain_database+(i*m_M+j)*2));
            else 
                index = (int)vec[j];
            sum += m_sub_distances[j][index];
        }
        if (max_heap.size() < top_k) {
            max_heap.push(make_pair(sum, i));
        } else if (sum < max_heap.top().first) {
            max_heap.pop();
            max_heap.push(make_pair(sum, i));
        }
//        if (sum < results.front().second) {
            // TODO this is wrong!!! The top element is not always on the top
//            pop_heap(results.begin(), results.end(), comp_max_heap);
//            results[top_k-1] = make_pair(i, sum);
/** Previous method for maintaining heap
            //print_heap(results);
            results[0] = make_pair(i, sum);
            //print_heap(results);
            make_heap(results.begin(), results.end(),comp_max_heap);
            //print_heap(results);
            //cout << endl;
            //if (i >= 10) exit(0);
*/
//        }
    }
    //print_heap(results);
    //int dist_m = 0;
    //vector<uchar> query_code = EncodePlain(query);
    //for (int i = 0; i <m_M; i ++) {
    //    if (plain_database[min_id*m_M+i] != query_code[i]) dist_m ++;
    //}
    //dist_hist[dist_m] ++;
    for (int i = top_k - 1; i >= 0; i --) {
        const pair<float, uint>& top = max_heap.top();
        results[i] = make_pair(top.second, top.first);
        max_heap.pop();
    }
    return results;
}
/** old plain query with only top 1
vector<pair<int, float> > PQTree::QueryPlain(const vector<float> &query, int top_k) {
    //cout << get_current_time_str() << endl;
    for (int i = 0; i < top_k; i ++) results[i].second = FLT_MAX;
    // calculate sub-distances
    for (int i = 0; i < m_M; i++) {
        for (int j = 0; j < m_Ks; j ++) {
            float dist_2 = 0;
            for (int k = 0; k < m_Ds; k ++) {
                float dist = m_codewords[i][j][k] - query[i*m_Ds+k];
                dist_2 += dist * dist;
            }
            m_sub_distances[i][j] = dist_2;
        }
    }

    float min_dist = FLT_MAX;
    long min_id = -1;
    float max_dist = 0;
    // scan the entire database
    for (long i = 0; i < N; i ++) {
        
        float sum = 0;
        for (int j = 0; j < m_M; j ++) {
            int index;
            if (PQ_K > 256) 
                index = (int)*((uint16_t*)(plain_database+(i*m_M+j)*2));
            else 
                index = (int)plain_database[i*m_M+j];
            sum += m_sub_distances[j][index];
        }
        if (sum < min_dist) {
            min_dist = sum;
            min_id = i;
        }
    }
    results[0] = make_pair(min_id, min_dist);
    int dist_m = 0;
    vector<uchar> query_code = EncodePlain(query);
    for (int i = 0; i <m_M; i ++) {
        if (plain_database[min_id*m_M+i] != query_code[i]) dist_m ++;
    }
    dist_hist[dist_m] ++;
    return results;
}
**/
#endif
// find the top_k nearest neighbors of query vector
vector<pair<int, float> > PQTree::Query(const vector<float> &query, int top_k) {
    #ifdef PLAIN_PQ
    //cout << "Query() called QueryPlain()" << endl;
    return QueryPlain(query, top_k);
    #endif
    n_Nodes = 0; // INFO number of inner nodes created
    
    vector<pair<int, float>> results(top_k);
    // calculate sub-distances
    for (int i = 0; i < m_M; i++) {
        for (int j = 0; j < m_Ks; j ++) {
            float dist_2 = 0;
            for (int k = 0; k < m_Ds; k ++) {
                float dist = m_codewords[i][j][k] - query[i*m_Ds+k];
                dist_2 += dist * dist;
            }
            m_sub_distances[i][j] = dist_2;
        }
    }
    
    #ifdef BOUNDS_TABLE
    // calculate prefix bounds
    for (int i = 0; i < m_M; i ++) {
        int pad = 1 << 1; // the first two elements in the array are paddings
        for (int level = (int)log2(m_Ks)-1; level > 0; level --) { // level [1,7]
            int length = 1 << level; // number of bounds in this level
            if (level == (int)log2(m_Ks)-1) { // check distance table
                //for (int idx = 0; idx < 256; idx ++) { // INFO checking
                //    cout << m_sub_distances[i][idx] << " ";
                //    if (idx%2==1) cout << ";   ";
                //}
                //cout << endl << endl;
                for (int item = 0; item < length; item ++) {
                    int min_idx = (length + item) << 1, max_idx = min_idx + 1;
                    float lwr_bnd = min(m_sub_distances[i][item<<1], 
                                        m_sub_distances[i][(item<<1)+1]);
                    float upr_bnd = max(m_sub_distances[i][item<<1],
                                        m_sub_distances[i][(item<<1)+1]);
                    m_sub_bounds[i][min_idx] = lwr_bnd;
                    m_sub_bounds[i][max_idx] = upr_bnd;
                }
                //for (int idx = 256; idx < 512; idx ++) { // INFO checking
                //    cout << m_sub_bounds[i][idx] << " ";
                //    if (idx%2==1) cout << ";   ";
                //}
                //cout << endl << endl << endl << endl;
            } else { // check bounds table from the level below
                for (int item = 0; item < length; item ++) {
                    int min_idx = (length + item) << 1, max_idx = min_idx + 1;
                    float lwr_bnd = min(m_sub_bounds[i][min_idx<<1], 
                                        m_sub_bounds[i][min_idx+1<<1]);
                    float upr_bnd = max(m_sub_bounds[i][(min_idx<<1)+1],
                                        m_sub_bounds[i][(min_idx+1<<1)+1]);
                    m_sub_bounds[i][min_idx] = lwr_bnd;
                    m_sub_bounds[i][max_idx] = upr_bnd;
                }
                //for (int idx = 0; idx < length*2; idx ++) { // INFO checking
                //    cout << m_sub_bounds[i][length*2+idx] << " ";
                //    if (idx%2==1) cout << ";   ";
                //}
                //cout << endl << endl << endl << endl;
            }
        }
        //exit(0); // INFO checking
    }
    #endif

    // find the nearest centroids (with centroid id and distance)
    //float debug = 0; // TODO DEBUG
    vector<pair<int, float>> closest_centroids(m_M);
    for (int i = 0 ; i < m_M; i ++) {
        float min_dist = numeric_limits<float>::max();
        int min_id = 0;
        for (int j = 0; j < m_Ks; j ++) {
            if (m_sub_distances[i][j] < min_dist) {
                min_dist = m_sub_distances[i][j];
                min_id = j;
            }
        }
        //debug+=min_dist; cout << bitset<8>(min_id) <<" ";// TODO DBEUG
        //for (int j = 0; j < m_Ds; j++) {
        //    cout << m_codewords[i][min_id][j] << " ";// TODO DEBUG
        //}
        //cout << endl;// TODO DBEUG
        closest_centroids[i] = make_pair(min_id, min_dist);
    }
    //cout << debug << endl; // TODO DEBUG
    // encode the query vector and find the closest vector in database
    bitset<CL> query_code = Encode(query);
    cout << query_code << endl;// TODO DEBUG
    //cout << query_code.to_ulong() << endl;// TODO DEBUG
    long index = search_prefix(query_code.to_ulong(), 0, database->Size());
    float coarse_dist = CalVectorDist(query, Decode(database->GetVec(index)));
    //coarse_dist = CalCoarseBound(query_code);
    // TODO DEBUG lines, please delete
    cout << "coarse distance : " << coarse_dist << endl; // DEBUG
    //cout << database->GetVec(index) << endl; // DEBUG
    //for (int i = 0; i < m_M * m_Ds; i ++)   // DEBUG
    //cout << query[i] <<" "<<Decode(database->GetVec(index))[i] << endl; // DEBUG
    //cout << get_current_time_str() << endl;
    //cout << "minimun distance is " << MinDist(query) << endl; // DEBUG
    //cout << get_current_time_str() << endl;
    // calculate upper bound and lower bound for the closest combination
    float upper_bound = 0, lower_bound = 0;
    for (int i = 0; i < m_M; i ++) {
        int part = (closest_centroids[i].first)/128;
        //cout << part << endl; // DEBUG
        float sub_upper = 0, sub_lower = numeric_limits<float>::max();
        #ifdef BOUNDS_TABLE
        // find bounds in first level of the bounds table
        int offset = ((1<<1) + part) << 1;
        lower_bound += m_sub_bounds[i][offset];
        upper_bound += m_sub_bounds[i][offset+1];
        #else
        for (int j = part*128; j < part*128+128; j++) {
            if (m_sub_distances[i][j] > sub_upper) sub_upper=m_sub_distances[i][j];
            if (m_sub_distances[i][j] < sub_lower) sub_lower=m_sub_distances[i][j];
        }
        upper_bound += sub_upper;
        lower_bound += sub_lower;
        #endif 
    }
    cout << "lower bound " << lower_bound << " upper bound " << upper_bound << endl;
    //return results;
    root = new Node();
    //float up_bnd = upper_bound;
    float up_bnd = coarse_dist;
    #ifdef SORTED_BOUND
    BuildTreeSorted(root, up_bnd);
    #else
    BuildTree(root, up_bnd);
    #endif
    cout << n_Nodes << endl;
    cout << up_bnd << endl;
    //cout << get_current_time_str() << endl;
    //
    float distance=numeric_limits<float>::max();
    int id=-1;
    ScanLeaves(root, up_bnd, distance, id);
    //cout << id << " " << distance << endl << endl;;
    results[0] = make_pair(id, distance);
    // TODO clear the tree and relevant data

    //cout << get_current_time_str() << endl;
    //cout << n_Nodes << endl;
    cout<<"leaf nodes " << n_vecs  << endl;
    n_vecs = 0;
    //exit(0);
    //TraverseTree(root);
    return results;
}
long PQTree::search_prefix(unsigned long prefix, long start, long end) {
    if (start == end) return start;
    if (start+1 == end) { 
        if (start+1 == database->Size()) return start;
        if (database->GetVec(start).to_ulong() < prefix && 
            database->GetVec(end).to_ulong() > prefix )
            return end;
    }
    long mid = start + (end-start)/2;
    //cout << start << " " << end << endl;
    //cout << mid << " "<< database->GetVec(mid).to_ulong() << " " << prefix << endl;
    if (database->GetVec(mid).to_ulong() < prefix) 
        return search_prefix(prefix,mid,end);
    else if (database->GetVec(mid).to_ulong() > prefix) 
        return search_prefix(prefix,start,mid);
    else {
        if (mid > 0 && database->GetVec(mid) == database->GetVec(mid-1)) {
            return search_prefix(prefix,start,mid);
        }
        return mid;
    }
}
void PQTree::BuildTree(Node* node, float& m_upper_bound){
    //cout << "BuilTree called on level " <<node->level<<" " <<node->size <<
    //    "-------------------------------------------------------" << endl;
    if (node->level == Node::max_level) {
        node->isLeaf = true;
        //cout << "8th level "<< node->prefix << endl;     // DEBUG
        return;
    }
    // handle each child
    int child_count = 0;
    for (int i = 0; i < node->size; i ++) {
        // search database to check if vectors with prefix exists
        bool exists_in_database;
        bitset<CL> child_prfx(node->prefix);
        int shift = (Node::max_level-1-node->level) * m_M;
        // construct child prefix (correctness checked)
        for (int j = 0; j < m_M; j ++) {
            bool val = (1<<j) & i;
            int idx = m_M-1-j+shift; // first bit should be the most significant bit
            child_prfx.set(idx, val);
        }
        // probe the database with prefix, then we can know if prefix exists
        long start_d = search_prefix(child_prfx.to_ulong(),0,database->Size());
        long end_d = search_prefix(child_prfx.to_ulong() + (((unsigned long)1) << shift),
                                    0, database->Size());
        exists_in_database = (end_d > start_d);
        // If prefix does not exists, check next child node
        float upper_bound = 0, lower_bound = 0;
        if (!exists_in_database) {
            node->upper_bounds[i] = upper_bound;
            node->lower_bounds[i] = lower_bound;
            //cout <<"    Not exists" << endl;
            continue;
        }
        int child_level = node->level+1;
        #ifdef BOUNDS_TABLE
        for (int j = 0; j < m_M; j ++) {
            int start=0;
            for (int k = 0; k < child_level; k ++) {
                // get bit from prefix
                int idx = CL - 1 - (k*m_M+j);
                bool bit = child_prfx.test(idx);
                if (k != 0) start = start << 1;
                start += bit;
            }
            if (child_level == Node::max_level) {
                lower_bound += m_sub_distances[j][start];
                upper_bound += m_sub_distances[j][start];
            } else {
                int offset = (1 << child_level) + start;
        //        cout << offset << endl;
                lower_bound += m_sub_bounds[j][offset<<1];
                upper_bound += m_sub_bounds[j][(offset<<1)+1];
            }
        }
        //cout << "level: "<< child_level << " "<< lower_bound << " " << upper_bound << " from table" << endl;
        //exit(0);
        #else
        // check upper bounds of this combination
        for (int j = 0; j < m_M; j ++) {
            // get the range of centroids in this sub dimension
            int start=0;
            for (int k = 0; k < child_level; k ++) {
                // get bit from prefix
                int idx = CL - 1 - (k*m_M+j);
                bool bit = child_prfx.test(idx);
                if (k != 0) start = start << 1;
                start += bit;
            }
            int end = start + 1;
            start = start << (Node::max_level - child_level);
            end = end << (Node::max_level - child_level);
            //DEBUG
            //if (start_d <= 905781430 && end_d >= 905781430 && (end_d-start_d)<=2) {
            //    cout<<"    In BuildTree() "<<start<<" "<<end<<endl;   // DEBUG
            //}
            // inspect the upper bounds and lower bounds
            float sub_upper = 0, sub_lower = numeric_limits<float>::max();
            for (int k = start; k < end; k++) {
                if (m_sub_distances[j][k] > sub_upper) sub_upper=m_sub_distances[j][k];
                if (m_sub_distances[j][k] < sub_lower) sub_lower=m_sub_distances[j][k];
            }
            upper_bound += sub_upper;
            lower_bound += sub_lower;
        }
        //if (child_level == 1)
        //    cout << "level: "<< child_level << " "<< lower_bound << " " << upper_bound << endl;
        #endif
        //cout << "Level "<<node->level << " s " << start_d <<" e " << end_d<<" shift "<<shift << endl;
        //cout << "Child prefix "<<child_prfx << endl;     // DEBUG
        //cout << exists_in_database << " " <<lower_bound <<" "<< upper_bound<<" "
        //     << m_upper_bound << endl;  // DEBUG
        //if (n_Nodes >= 2)exit(0);        // DEBUG

        node->upper_bounds[i] = upper_bound;
        node->lower_bounds[i] = lower_bound;
        
        if (upper_bound < m_upper_bound) m_upper_bound = upper_bound;
        //if (node->level == 0) continue;
        //if (child_level == 1)
        //    cout << child_level<< " "<< lower_bound << " " << m_upper_bound << " "
        //        <<(abs(lower_bound - m_upper_bound) < m_upper_bound*EPS) << endl;
        if (exists_in_database && 
                ((abs(lower_bound- m_upper_bound) < m_upper_bound*EPS) 
                  || lower_bound < m_upper_bound) ) {
            // build the sub tree
            node->children[i] = new Node(node->level+1, child_prfx);
            BuildTree(node->children[i], m_upper_bound);
            n_Nodes ++;
            child_count ++;
        }
    }
    if (child_count == 0) node->isLeaf = true;
    //cout << "End of BuilTree() calling ====================="<< endl;
    //exit(0);
}

void PQTree::BuildTreeSorted(Node* node, float& m_upper_bound){
    if (node->level == Node::max_level) {
        node->isLeaf = true;
        //cout << "8th level "<< node->prefix << endl;     // DEBUG
        return;
    }
    vector<bitset<CL> > child_prfxs(node->size);
    // handle each child
    int child_count = 0;
    vector<pair<int, float>> sorted_lwrbnd(node->size);
    for (int i = 0; i < node->size; i ++) {
        // search database to check if vectors with prefix exists
        bool exists_in_database;
        bitset<CL> child_prfx(node->prefix);
        int shift = (Node::max_level-1-node->level) * m_M;
        // construct child prefix (correctness checked)
        for (int j = 0; j < m_M; j ++) {
            bool val = (1<<j) & i;
            int idx = m_M-1-j+shift; // first bit should be the most significant bit
            child_prfx.set(idx, val);
        }
        child_prfxs[i] = child_prfx;
        // probe the database with prefix, then we can know if prefix exists
        long start_d = search_prefix(child_prfx.to_ulong(),0,database->Size());
        long end_d = search_prefix(child_prfx.to_ulong() + (((unsigned long)1) << shift),
                                    0, database->Size());
        exists_in_database = (end_d > start_d);
        // If prefix does not exists, check next child node
        float upper_bound = 0, lower_bound = 0;
        if (!exists_in_database) {
            node->upper_bounds[i] = upper_bound;
            node->lower_bounds[i] = FLT_MAX;
            sorted_lwrbnd[i] = make_pair(i, FLT_MAX);
            continue;
        }
        int child_level = node->level+1;
        #ifdef BOUNDS_TABLE
        for (int j = 0; j < m_M; j ++) {
            int start=0;
            for (int k = 0; k < child_level; k ++) {
                // get bit from prefix
                int idx = CL - 1 - (k*m_M+j);
                bool bit = child_prfx.test(idx);
                if (k != 0) start = start << 1;
                start += bit;
            }
            if (child_level == Node::max_level) {
                lower_bound += m_sub_distances[j][start];
                upper_bound += m_sub_distances[j][start];
            } else {
                int offset = (1 << child_level) + start;
                lower_bound += m_sub_bounds[j][offset<<1];
                upper_bound += m_sub_bounds[j][(offset<<1)+1];
            }
        }
        #else
        // check upper bounds of this combination
        for (int j = 0; j < m_M; j ++) {
            // get the range of centroids in this sub dimension
            int start=0;
            for (int k = 0; k < child_level; k ++) {
                // get bit from prefix
                int idx = CL - 1 - (k*m_M+j);
                bool bit = child_prfx.test(idx);
                if (k != 0) start = start << 1;
                start += bit;
            }
            int end = start + 1;
            start = start << (Node::max_level - child_level);
            end = end << (Node::max_level - child_level);
            // inspect the upper bounds and lower bounds
            float sub_upper = 0, sub_lower = numeric_limits<float>::max();
            for (int k = start; k < end; k++) {
                if (m_sub_distances[j][k] > sub_upper) sub_upper=m_sub_distances[j][k];
                if (m_sub_distances[j][k] < sub_lower) sub_lower=m_sub_distances[j][k];
            }
            upper_bound += sub_upper;
            lower_bound += sub_lower;
        }
        #endif
        node->upper_bounds[i] = upper_bound;
        node->lower_bounds[i] = lower_bound;
        sorted_lwrbnd[i] = make_pair(i, lower_bound);
    }
    sort(sorted_lwrbnd.begin(), sorted_lwrbnd.end(), bound_sorter);
    for (int idx = 0; idx < node->size; idx ++) {
    //    cout <<idx<<" "<<sorted_lwrbnd[idx].first << " " << sorted_lwrbnd[idx].second
    //            << endl;
    
        int i = sorted_lwrbnd[idx].first;
        float upper_bound = node->upper_bounds[i];
        float lower_bound = node->lower_bounds[i];
        if (lower_bound == FLT_MAX) continue;
        if (upper_bound < m_upper_bound) m_upper_bound = upper_bound;
        //if (node->level == 0) continue;
        //if (child_level == 1)
        //    cout << child_level<< " "<< lower_bound << " " << m_upper_bound << " "
        //        <<(abs(lower_bound - m_upper_bound) < m_upper_bound*EPS) << endl;
        if ( ((abs(lower_bound- m_upper_bound) < m_upper_bound*EPS) 
                  || lower_bound < m_upper_bound) ) {
            // build the sub tree
            node->children[i] = new Node(node->level+1, child_prfxs[i]);
            BuildTreeSorted(node->children[i], m_upper_bound);
            n_Nodes ++;
            //if (n_Nodes % 10000 == 0) cout <<node->level+1<<" " << lower_bound << " " << upper_bound<< " "<<m_upper_bound << endl;
            child_count ++;
        }
    }
    if (child_count == 0) node->isLeaf = true;
}
void PQTree::ScanLeaves(Node* node, const float bound, float& dist, int& id) {
    //n_Nodes--;
    if (node->level == Node::max_level) {
        // scan all the vectors under the node's prefix
        unsigned long start_elem = node->prefix.to_ulong();
        long start = search_prefix(node->prefix.to_ulong(),0,database->Size());
        unsigned long end_elem = start_elem + (1<<((Node::max_level-node->level)*m_M));
        long end = search_prefix(end_elem, 0, database->Size());
        //cout << start << endl << end << endl; // DEBUG
        n_vecs += 1;
        //for (long i = start; i < end; i ++) {
            long i = start;
            float distance = CalCodeDist(database->GetVec(i));
            if (distance < dist) {
                dist = distance;
                id = i;
            }
        //}
    } else {
        // check each valid child of the node
        for (int i = 0; i < node->size; i ++) {
            if (node->children[i] != NULL) {
                if ((node->lower_bounds[i] - bound) < bound*EPS)
                    ScanLeaves(node->children[i],bound, dist, id);
            }
        }
    }
}

PQTree::Node::Node() {
    level = 0;
    prefix = 0;
    size = 1 << PQ_M;
    children = new Node*[size];
    for (int i = 0; i < size; i ++) children[i]=NULL;
    upper_bounds = new float[size];
    lower_bounds = new float[size];
    memset(lower_bounds, 0, sizeof(float)*size);
    memset(upper_bounds, 0, sizeof(float)*size);
}

PQTree::Node::Node(int level, bitset<CL> prfx) {
    size = 1 << PQ_M;
    children = new Node*[size];
    for (int i = 0; i < size; i ++) children[i]=NULL;
    upper_bounds = new float[size];
    lower_bounds = new float[size];
    memset(lower_bounds, 0, sizeof(float)*size);
    memset(upper_bounds, 0, sizeof(float)*size);
    this->level = level;
    this->prefix = bitset<CL>(prfx);
}

PQTree::Node::~Node() {
    delete[] children;
    delete[] upper_bounds;
    delete[] lower_bounds;
}

void PQTree::Write(string path, const uchar* vecs, long N)
{
    ofstream ofs(path, ios::binary);
    if(!ofs.is_open()){
        cerr << "Error: cannot open " << path << ends;
        assert(0);
    }

    // Write (1) N, (2) D, and (3) data
    ofs.write( reinterpret_cast<char*> (&N), sizeof(long));
    cout << "N = "<<N << endl;
    //for(long n = 0; n < N*m_M; ++n){
    //    ofs.write( (char *) &(vecs[n]), sizeof(uchar));
    //}
    // The other way to write
    if (PQ_K > 256)
        ofs.write( (char*) &(vecs[0]), sizeof(uchar)*N*PQ_M*2);
    else
        ofs.write( (char*) &(vecs[0]), sizeof(uchar)*N*PQ_M);
    ofs.close();
}
void PQTree::Read(string path, uchar*& vecs, long& N, int top_n)
{
    ifstream ifs(path, ios::binary);
    if(!ifs.is_open()){
        cout << path << endl;
        cerr << "Error: cannot open " << path << ends;
        assert(0);
    }

    // Read (1) N, (2) D, and (3) data
    ifs.read(reinterpret_cast<char*> (&N), sizeof(long));
    cout << "Read: N = " << N << endl;

    if(top_n == -1){
        top_n = N;
    }
    assert(0 < top_n && top_n <= N);
    cout << "database vecs length is " << N*PQ_M<< endl;
    if (PQ_K > 256) {
        if (with_id) {
            cout << "K>256 with id is not implemented yet" << endl;
            exit(0);
        }
        vecs = new uchar[N*PQ_M*2];
        memset(vecs, 0, sizeof(uchar)*(N*PQ_M)*2);
    } else {
        if (with_id) {
            vecs = new uchar[N*(PQ_M+sizeof(int))];
            memset(vecs, 0, sizeof(uchar)*(N*(PQ_M+sizeof(int))));
        } else {
            vecs = new uchar[N*PQ_M];
            memset(vecs, 0, sizeof(uchar)*(N*PQ_M));
        }
    }
    cout << "start to read from file : " << endl << "   " 
        << path << endl;
    //for(long n = 0; n < top_n*m_M; ++n) {
    //    ifs.read( (char *) &vecs[n], sizeof(uchar));
    //}
    // The other way to read
    if (PQ_K > 256)
        ifs.read( (char *) &(vecs[0]), sizeof(uchar)*N*PQ_M*2);
    else {
        if (with_id) 
            ifs.read( (char *) &(vecs[0]), sizeof(uchar)*N*(PQ_M+sizeof(int)));
        else           
            ifs.read( (char *) &(vecs[0]), sizeof(uchar)*N*PQ_M);
    }
    ifs.close();
}
void PQTree::Node::CalBounds(const vector<vector<float>>& sub_distances) {
    
}

BitVecs::BitVecs(const vector<bitset<CL> > &vec)
{
    assert(!vec.empty());
    Resize((int) vec.size(), (int) vec[0].size());
    for(int n = 0; n < m_N; ++n){
        SetVec(n, vec[n]);
    }
}

void BitVecs::Resize(int N, int D)
{
    m_N = N;
    m_D = D;
    //m_data.clear(); // values are remained
    m_data.resize( (unsigned long long) m_N );
}

const bool BitVecs::GetVal(int n, int d) const {
    //assert(0 <= n && n < m_N && 0 <= d && d < m_D);
    return m_data[n][d];
}

bitset<CL> BitVecs::GetVec(int n) const {
    assert(0 <= n && n < m_N);
    //return bitset<CL>(m_data[n]);
    return (m_data[n]);
}


void BitVecs::SetVal(int n, int d, bool val){
    if (!(0 <= n && n < m_N && 0 <= d && d < m_D)) {
        cout << "n = " <<n << " m_N = "<< m_N << " d = " << d << " m_D = "<< m_D << endl;
    }
    assert(0 <= n && n < m_N && 0 <= d && d < m_D);
    m_data[ (unsigned long long) n][d] = val;
}

void BitVecs::SetVec(int n, const bitset<CL> vec)
{
    assert(0 <= n && n < m_N);
    m_data[n] = vec;
//    for(int d = 0; d < CL; ++d){
//        SetVal(n, d, vec[d]);
//    }
}

void BitVecs::Write(string path, const BitVecs &vecs)
{
    ofstream ofs(path, ios::binary);
    if(!ofs.is_open()){
        cerr << "Error: cannot open " << path << ends;
        assert(0);
    }

    // Write (1) N, (2) D, and (3) data
    int N = vecs.Size();
    int D = vecs.Dim();
    ofs.write( (char *) &N, sizeof(int));
    ofs.write( (char *) &D, sizeof(int));
    for(int n = 0; n < N; ++n){
        bitset<CL> vec = vecs.GetVec(n);
        ofs.write( (char *) &vec, sizeof(bitset<CL>));
    }
    // The other way to write
    //ofs.write( (char*) &(vecs.m_data[0]), sizeof(bitset<CL>)*N );
    ofs.close();
}

void BitVecs::Read(string path, BitVecs *vecs, int top_n)
{
    assert(vecs != NULL);
    ifstream ifs(path, ios::binary);
    if(!ifs.is_open()){
        cerr << "Error: cannot open " << path << ends;
        assert(0);
    }

    // Read (1) N, (2) D, and (3) data
    int N, D;
    ifs.read( (char *) &N, sizeof(int));
    ifs.read( (char *) &D, sizeof(int));
    cout << "N = " << N << endl;
    cout << "D = " << D << endl;

    if(top_n == -1){
        top_n = N;
    }
    assert(0 < top_n && top_n <= N);

    vecs->Resize(top_n, D);
    cout << "start to read" << endl;
    for(int n = 0; n < top_n; ++n){
        bitset<CL> buf(D);
        ifs.read( (char *) &buf, sizeof(bitset<CL>));
        vecs->SetVec(n, buf);
    }
    // The other way to read
    //ifs.read( (char *) &(vecs->m_data[0]), sizeof(bitset<CL>)*N );
}

BitVecs BitVecs::Read(string path, int top_n)
{
    BitVecs codes;
    Read(path, &codes, top_n);
    return codes;
}

void BitVecs::SortVecs() {
    sort(m_data.begin(), m_data.end(), bitset_sorter);
    for (int i = 0; i < 20; i ++) {
        cout << m_data[i] << endl;
    }
}
inline float PQTree::CalVectorDist(const vector<float> & a,const vector<float> &b){
    float sum = 0;
    assert( a.size() == b.size() );
    for (int i = 0; i < a.size(); i ++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}
inline float PQTree::CalCodeDist(const bitset<CL> code){
    float sum = 0;
    for (int i = 0; i < m_M; i ++) {
        int index = 0;
        for (int j = 0; j < CL/m_M; j ++) {
            int idx = CL-1-(i+j*m_M);
            //int idx = (m_M-1-i) + (CL/m_M-1-j)*m_M;
            index += code.test(idx);
            if (j != CL/m_M-1)
                index <<= 1;
        }
        sum += m_sub_distances[i][index];
    }
    return sum;
}
vector<float> PQTree::Decode(const bitset<CL> code) {
    vector<float> vec;
    //vec.reserve(m_M * m_Ds);
    for (int i = 0; i < m_M; i ++) {
        int index = 0;
        for (int j = 0; j < CL/m_M; j ++) {
            int idx = CL-1-(i+j*m_M);
            index += code.test(idx);
            if (j != CL/m_M-1)
                index <<= 1;
        }
        for (int j = 0; j < m_Ds; j ++) {
            float value = m_codewords[i][index][j];
            vec.push_back(value);
//            cout << value << " ";
        }
//        cout << endl << endl;
    }
//    cout << code << endl;
    return vec;
}
vector<float> PQTree::DecodePlain(const uchar* code) {
    vector<float> vec;
    //vec.reserve(m_M * m_Ds);
    for (int i = 0; i < m_M; i ++) {
        int index = (int)code[i];
        for (int j = 0; j < m_Ds; j ++) {
            float value = m_codewords[i][index][j];
            vec.push_back(value);
//            cout << value << " ";
        }
//        cout << endl << endl;
    }
//    cout << code << endl;
    return vec;
}
float PQTree::MinDist(const vector<float> query) {
    //cout << endl;//DEBUG
    float min_dist = numeric_limits<float>::max();
    float max_dist = 0;
    int min_id = -1;
    for (long i = 0; i < database->Size(); i ++) {
//        vector<float> vec = Decode(database->GetVec(i));
//        float dist = CalVectorDist(query, vec);
        float dist = CalCodeDist(database->GetVec(i));
        if (min_dist > dist) {
            min_id = i;
            min_dist = dist;
        }
        if (max_dist < dist) max_dist = dist;
    }
    cout << " " << max_dist <<" ";
    cout << min_id << " ";// DEBUG
    return min_dist;
}
float PQTree::CalCoarseBound(const bitset<CL> prefix) {
    int level = 0;
    bitset<CL> tmp_prfx(0);
    for (; level < Node::max_level; level++) {
        int shift = (Node::max_level-1-level) * m_M;
        for (int m = 0; m <m_M; m ++) {
            int idx = m_M-1-m+shift;
            tmp_prfx.set(idx, prefix.test(idx));
        }
        long start_d = search_prefix(tmp_prfx.to_ulong(), 0, database->Size());
        long end_d =   search_prefix(tmp_prfx.to_ulong() + 
                                    (((unsigned long)1) << shift),
                                    0, database->Size());
        //cout << tmp_prfx << endl;
        // check existence
        if (start_d == end_d) break;
    }
    float upper_bound;
    for (int j = 0; j < m_M; j ++) {
        int start=0;
        for (int k = 0; k < level; k ++) {
            // get bit from prefix
            int idx = CL - 1 - (k*m_M+j);
            bool bit = tmp_prfx.test(idx);
            if (k != 0) start = start << 1;
            start += bit;
        }
        if (level == Node::max_level) {
            upper_bound += m_sub_distances[j][start];
        } else {
            int offset = (1 << level) + start;
            //cout << j <<" offset: "<< offset << " " << start << endl;
            upper_bound += m_sub_bounds[j][(offset<<1)+1];
        }
    }
    //exit(0);
    return upper_bound;
}
