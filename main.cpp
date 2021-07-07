#include <omp.h>
#include <pthread.h>
#include "pq.h"
#include "pq_tree.h"
#include "pqbase.h"
#include "utils.h"
#include <unistd.h>


using namespace std;

// important variables
int PQ_K = 256;
int PQ_M = 8;
int with_id = 0;

bool file_exists_test(const std::string &name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }
}

#ifdef PLAIN_PQ
vector<uchar> EncodeParallel(PQTree &pqtree, 
                                    vector<vector<float> > &vecs){

    vector<uchar > codes;
    if (PQ_K > 256)
        codes.resize(vecs.size()*PQ_M*2);
    else 
        codes.resize(vecs.size()*PQ_M);
    //omp_set_num_threads(40);
    #ifndef ENCODE_DEBUG
    //cout << "plain parallel for called" << endl;
    #pragma omp parallel for
    #endif
    for(int i = 0; i < (int) vecs.size(); ++i){
        #ifdef QUERY_DEBUG
        i = vecs.size() - 1;
        cout << "check i =" << i << endl;
        #endif
        vector<uchar> one_code = pqtree.EncodePlain(vecs[i]);
        for (int j = 0; j < PQ_M; j ++) {
            if (PQ_K > 256) {
                codes[(i*PQ_M+j)*2] = one_code[j*2];
                codes[(i*PQ_M+j)*2+1]=one_code[j*2+1];
            } else {
                codes[i*PQ_M + j] = one_code[j];
            }
        }
    }
    return codes;
}
void Assign(uchar* base, long start_id, const vector<uchar> &added){
    //assert(start_id*PQ_M+ added.size() <= base.size());
    for(long i = 0; i < added.size(); ++i){
        if(PQ_K > 256) {
            int idx = added[i]*256 + added[i+1];
            ((uint16_t*)base)[start_id*PQ_M+i/2] = (uint16_t)idx;
            i++;
        } else {
            base[start_id*PQ_M + i] = added[i];
        }
    }
}

#else
// Encode vecs in parallel
vector<bitset<CL> > EncodeParallel(const PQTree &pqtree, 
                                    const vector<vector<float> > &vecs){
    cout << "parallel for called" << endl;
    vector<bitset<CL> > codes(vecs.size());
    #ifndef ENCODE_DEBUG
    #pragma omp parallel for
    #endif
    for(long i = 0; i <  vecs.size(); ++i){
        #ifdef ENCODE_DEBUG
        i = 11114457;
        #endif
        codes[i] = pqtree.Encode(vecs[i]);
    }
    return codes;
}
// Assign "added" to "base[start_id] - base[start_id + added.Size()"]
void Assign(BitVecs &base, long start_id, const BitVecs &added){
    assert(start_id + added.Size() <= base.Size());
    for(int i = 0; i < added.Size(); ++i){
        base.SetVec(start_id + i, added.GetVec(i));
    }
}
#endif

auto comp_max_heap = [](const pair<uint, float> &a, const pair<uint, float> &b) {
    return a.second < b.second;
};
void print_heap(vector<pair<uint, float>>& test) {
    for (int i = 0; i < test.size(); i ++) {
        cout << test[i].first << ", " << test[i].second << "    ";
    }
    cout << endl;
}
void batch_partial_topk_queries(vector<vector<float>>& buff, 
                                 vector<vector<float>>& queries, 
                                 vector<vector<pair<uint, float>>>& results,
                                 long long start_vec_id) {
    #pragma omp parallel for
    for (int i = 0; i < queries.size(); i ++) {
    // process each raw database vector
        vector<float>& query = queries[i];
        // calculate distance 
        for (long long it = 0; it < buff.size(); it ++) {
            auto& vec = buff[it];
            double distance=0.0;
            for (int d = 0; d < query.size(); d ++) {
                //distance += pow(vec[d]-query[d], 2);
                distance += (vec[d]-query[d])*(vec[d]-query[d]);
            }
            if (distance < results[i].front().second) {
                //print_heap(results[i]);
                results[i][0] = make_pair(start_vec_id+it, distance);
                //print_heap(results[i]);
                make_heap(results[i].begin(), results[i].end(), comp_max_heap);
                //print_heap(results[i]);
                //cout << endl;
                //if (it == 10) exit(0);
            }
        }
    }
}
auto cmp_max = [](pair<float, uint>& left, pair<float, uint>& right) {
    return left.first < right.first;
};
void batch_partial_topk_queries(vector<vector<float>>& buff, 
                                 vector<vector<float>>& queries, 
                                 vector<priority_queue<pair<float, uint>, 
                                        vector<pair<float, uint>>, 
                                        decltype(cmp_max)>>& results,
                                 int topk,
                                 long long start_vec_id) {
    #pragma omp parallel for
    for (int i = 0; i < queries.size(); i ++) {
    // process each raw database vector
        vector<float>& query = queries[i];
        // calculate distance 
        for (long long it = 0; it < buff.size(); it ++) {
            auto& vec = buff[it];
            double distance=0.0;
            for (int d = 0; d < query.size(); d ++) {
                //distance += pow(vec[d]-query[d], 2);
                distance += (vec[d]-query[d])*(vec[d]-query[d]);
            }
            if (results[i].size() < topk) {
                results[i].push(make_pair(distance, start_vec_id+it));
            } else if (distance < results[i].top().first) {
                results[i].pop();
                results[i].push(make_pair(distance, start_vec_id+it));
            }
        }
    }
    
}
    
int main(int argc, char* argv[]){
    string dataset;
    string queryset;
    string grndtruth;
    string ext = "fvecs";
    int train_size=-1;
    string task = "learn";
    int top_k = 1;
    int query_size = -1;
    int debug = 0;
    int random_sample = 0;
    int synth = 0;
    long NN=-1;
    int reordered_cid = 0;
    int uni_sample=0;
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") {
    		dataset = string(argv[i + 1]);
        }
        if (arg == "-queryset") {
    		queryset = string(argv[i + 1]);
        }
        if (arg == "-grndtruth") {
    		grndtruth = string(argv[i + 1]);
        }
        if (arg == "-ext") {
    		ext = string(argv[i + 1]);
        }
        if (arg == "-task") {
            task = string(argv[i + 1]);
        }
        if (arg == "-topk") {
            top_k = atoi(argv[i + 1]);
        }
        if (arg == "-query_size") {
            query_size = atoi(argv[i+1]);
        }
        if (arg == "-train_size") {
            train_size = atoi(argv[i+1]);
        }
        if (arg == "-k") {
            PQ_K = atoi(argv[i+1]);
        }
        if (arg == "-m") {
            PQ_M = atoi(argv[i+1]);
        }
        if (arg == "-debug") {
            debug = 1;
        }
        if (arg == "-rand_sample") {
            random_sample = 1;
        }
        if (arg == "-synth" ) {
            synth = 1;
        }
        if (arg == "-N") {
            NN = atol(argv[i + 1]);
        }
        if (arg == "-r") {
            reordered_cid = 1;
        }
        if (arg == "-us") { // uniform sample of base vectors for training
            uni_sample = 1;
        }   
    }
    char* name = get_current_dir_name();
    cout << "Current Working Dir: " << name << "\n";
    if (task == "learn") {
        // (1) Make sure you have already downloaded sift1b data in data/ 
        //     by scripts/download_siftsmall.sh

        // (2) Read vectors
        int top_n = train_size; // Use top top_n vectors to train codebook
        cout << "train size: " << train_size << endl;
        cout <<get_current_time_str() << endl;
        //vector<vector<float> > learns = 
        //        ReadTopN("../../data/bigann_learn.bvecs", "bvecs", top_n);
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << dataset << endl;
        if (NN != -1) cout << "N = " << NN << endl;
        long load_size = 100000;  // perform random shuffle and resize before training
        string learn_set = dataset+"/learn."+ext;
        if (uni_sample == 1) learn_set = learn_set + ".unisample";
        vector<vector<float> > learns = ReadTopN( learn_set, ext, load_size);
//        for (int i = 0; i < 2; i ++) {
//            for (int j = 0; j < learns[i].size() ; j ++) {
//                cout << learns[i][j] << " ";
//            }
//            cout << endl;
//        }
        cout << "all vectors " << learns.size() << endl;
        //vector<vector<float> > learns = ReadTopN(dataset, ext, -1);
        __gnu_parallel::random_shuffle(learns.begin(), learns.end());
        if (train_size != -1)
            learns.resize(train_size);
        cout << "vectors read" << endl;

        // (3) Train a product quantizer
        timeval beg, mid, mid1, end, all_st, all_en; 
        gettimeofday(&all_st, NULL);
        int M =PQ_M;  // You can change this to 8. M=4: a PQ-code is 32-bit.  M=8: a PQ-code is 64-bit.
        cout << "=== Train a product quantizer ===" << endl;
        vector<PQ::Array> codewords = PQ::Learn(learns, M, PQ_K);
        string filepath = dataset + "/M" + to_string(M) + "K"+to_string(PQ_K)
                            +"codewords.txt";
        PQ::WriteCodewords(filepath, codewords);
        cout <<get_current_time_str() << endl;

        gettimeofday(&all_en, NULL);
        cout << "   Learning codebook uses: "
            << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
            << " sec" <<endl;
    }
    if (task == "combine") {
        cout << "M = " << PQ_M << endl;
        // (1) Make sure you've already run "demo_sift1b_train", and "codewords.txt" is in the bin dir.
        // (2) Setup a product quantizer
        char* name = get_current_dir_name();
        cout << "Plain naive PQ encoding" << endl;

        uchar* vecs1;
        long N1;
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+to_string(PQ_K), vecs1, N1);
        N1 = 894910897;
        cout << "N1 = " << N1 << endl;

        uchar* vecs2;
        long N2;
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+to_string(PQ_K)+".vali", vecs2, N2);
        cout << "N2 = " << N2 << endl;

        long N = 1000000000;
        uchar* vecs = new uchar[N*PQ_M];
        memcpy(vecs, vecs1, sizeof(uchar)*N1*PQ_M);
        memcpy(vecs+N1*PQ_M, vecs2, sizeof(uchar)*(N-N1)*PQ_M);

        cout <<get_current_time_str() << endl;
        // (5) Write codes
        //PQTree::Write("/media/bigdata/uqrwan14/vecs/data/bigann/codes.bin.plain", codes,N);
        PQTree::Write(dataset + "/codes.bin.plain.M"+to_string(PQ_M)
                        +"K"+to_string(PQ_K), vecs,N);
        cout << N << endl;
        cout << "database codes written "<<get_current_time_str() << endl;
    }
    if (task == "encode") {
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        // (1) Make sure you've already run "demo_sift1b_train", and "codewords.txt" is in the bin dir.
        // (2) Setup a product quantizer
        char* name = get_current_dir_name();
        #if defined(PLAIN_PQ) || defined(ASSIGN_MODULE)
            #ifdef ASSIGN_MODULE
            cout << "Testing assignment" << endl;
            #else
            cout << "Plain naive PQ encoding" << endl;
            #endif

        PQ pq(PQ::ReadCodewords(dataset+"/M"+to_string(PQ_M)+ "K"+to_string(PQ_K)
                                +"codewords.txt"));
        PQTree pqtree(pq.GetCodewords());
        #else
        PQ pq(PQ::ReadCodewords(dataset+"/M"+to_string(PQ_M)+ "K"+to_string(PQ_K)
                                +"codewords.txt"));
        PQTree pqtree(pq.GetCodewords());
        pqtree.DichotomizeCodewords(0, pq.GetKs()); // This is a recursive function
        PQ::WriteCodewords(dataset+"/M"+to_string(PQ_M)+ "K"+to_string(PQ_K)
                            +"relocated_codewords.txt", pqtree.GetCodewordsT());
        #endif
        // (3) Setup pqcodes
        long N = 1500000000;
        if (NN != -1) N = NN;
        cout << "N " << N << endl;
        
        #ifdef PLAIN_PQ
        uchar* codes;
        if (PQ_K > 256) {
            codes = new uchar[(long long)N*PQ_M*2];
        } else {
            codes = new uchar[(long long)N*PQ_M];
        }
        #else
        BitVecs codes(N, pq.GetDs()*pq.GetM());  // a PQ-code is D-dim bitset<>.
        #endif

        ItrReader reader(dataset+"/base."+ext, ext);
        //ItrReader reader(dataset+"/total."+ext, ext);
        vector<vector<float> > buff;  // Buffer
        long long  buff_max_size = N / 2000;
        buff_max_size = buff_max_size > 10000 ? buff_max_size : 10000;
        long long id_encoded = 0;

        int debug = 0;

        // timer
        timeval beg, mid, mid1, end, all_st, all_en; 
        gettimeofday(&all_st, NULL);

        long long n_codes = 0;
        cout << "Start encoding" <<get_current_time_str() << endl;
        while(!reader.IsEnd()){
            buff.push_back(reader.Next());  // Read a line (a vector) into the buffer
            if( (int) buff.size() == buff_max_size){  // (1) If buff_max_size vectors are read,
                Assign(codes, id_encoded, EncodeParallel(pqtree, buff)); // (2) Encode the buffer, and assign to the codes
                id_encoded += (int) buff.size();  // update id
                n_codes += (int) buff.size();
                buff.clear(); // (3) refresh
                //printf("\r%lld/%lld  %.1f%%     ",id_encoded, N, ((double)id_encoded)*100/N);
                fflush(stdout);
                //if (id_encoded/buff_max_size % 50 == 0)
                if (id_encoded % (N / 5) == 0)
                    cout << id_encoded << " / " << N << " vectors are encoded in total" << endl;
            }
            if (id_encoded == N) break;
        }
        cout << "debug " <<debug<< endl;
        if(0 < (int) buff.size()){ // Rest
            Assign(codes, id_encoded, EncodeParallel(pqtree, buff)); // Encode buff, and assign to codes
            n_codes += (int) buff.size();
            buff.clear();
        }
        cout << n_codes << endl;
        N = n_codes;
        cout <<get_current_time_str() << endl;
//        // INFO
//        for (int i = 0; i < 10; i ++) {
//            for (int m = 0; m < PQ_M; m ++) {
//                if (PQ_K > 256) {
//                    int val = ((uint16_t*)codes)[(N-1-i)*PQ_M+m];
//                    cout << val << " ";
//                } else {
//                    cout << (int)codes[(N-1-i)*PQ_M+m] << " ";
//                }
//            }
//            cout << endl;
//        }
        // INFO
        // (5) Write codes
        #ifdef PLAIN_PQ
        //PQTree::Write("/media/bigdata/uqrwan14/vecs/data/bigann/codes.bin.plain", codes,N);
        PQTree::Write(dataset + "/codes.bin.plain.M"+to_string(PQ_M)
                        +"K"+to_string(PQ_K) + "N" + to_string(N), codes,N);
        #elif defined(ASSIGN_MODULE) 
        // without clustering centoids
        codes.SortVecs();
        //BitVecs::Write("/media/bigdata/uqrwan14/vecs/data/bigann/codes.bin.assign0", codes);
        BitVecs::Write(dataset + "/codes.bin.assign0", codes);
        #else
        cout << "writing codes M = " <<  PQ_M << endl;
        BitVecs::Write(dataset + "/codes.bin.M"+to_string(PQ_M)+"K"+to_string(PQ_K), codes);
        #endif
        gettimeofday(&all_en, NULL);
        cout << "   Encoding uses: "
            << all_en.tv_sec - all_st.tv_sec + (all_en.tv_usec-all_st.tv_usec)/1e6
            << " sec" <<endl;
        cout << "database codes written "<<get_current_time_str() << endl;
    }
    if (task == "query") {
        cout <<"query" <<get_current_time_str()<<endl;
        // (1) Make sure you've already trained codewords and encoded database vectors
        // (2) Read query vectors
        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        
        #ifdef PLAIN_PQ
        cout << "Plain query" << endl;
        uchar* vecs;
        long N;
        if (reordered_cid == 0) {
            PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)
                            +"K"+to_string(PQ_K), vecs, N);
        } else if (reordered_cid == 1) {
            PQTree::Read(dataset+"/codes.bin.reordered.M"+to_string(PQ_M)
                            +"K"+to_string(PQ_K), vecs, N);
        }
        if (synth == 1){
            // generate first vector
            cout << "Generating random dataset" << endl;
            for (int i = 0; i < PQ_M; i ++) {
                vecs[i] = (uchar)(rand() % PQ_K);
            }
            for (int i = 1; i < N; i++) {
                for (int m = 0; m < PQ_M; m ++) vecs[i*PQ_M+m] = vecs[(i-1)*PQ_M+m];
                int selected_m = rand()%PQ_M;
                int k = rand() % PQ_K;
                int val = (int) vecs[i * PQ_M + selected_m];
                while (k == val) k = rand() % PQ_K;
                vecs[i*PQ_M+selected_m] = k;

//                if (i < 10) {
//                    for (int m = 0; m < PQ_M; m ++) cout << (int)vecs[i*PQ_M+m] << " ";
//                    cout << endl;
//                }
            }
        }
        if (NN != -1) N = NN;
        cout << "N = " << N << endl;
        string codewords_file_name;
        if (reordered_cid == 0) {
            codewords_file_name = dataset + "/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt";
        } else if (reordered_cid == 1) {
            codewords_file_name = dataset + "/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"relocated_codewords.txt";
        }
        PQ pq(PQ::ReadCodewords(codewords_file_name));
        PQTree pqtree(pq.GetCodewords(), vecs, N);
        // sample query vectors from dataset
        if (query_size != -1 && random_sample) {
            int interval = N/query_size;
            queries = vector<vector<float>>();
            for (long long i = 0; i < N; i += interval) {
                vector<float> query = pqtree.DecodePlain(vecs+i*PQ_M);
                queries.push_back(query);
            }
        }
        #elif defined(ASSIGN_MODULE)
        
        cout << "Test assignment module" << endl;
        BitVecs bitvecs;
        BitVecs::Read(dataset+"/codes.bin.assign0", &bitvecs);
        cout << "dataset read "<< get_current_time_str() << endl;
        PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+"K"+to_string(PQ_K)+"codewords.txt"));
        PQTree pqtree(pq.GetCodewords(), &bitvecs);

        #else

        // (3) Read database vectors
        BitVecs bitvecs;
        if (file_exists_test(dataset+"/codes.bin.M"+to_string(PQ_M)+ "K"+to_string(PQ_K)+".sorted")) {
            cout << "sorted database exists" << endl;
            BitVecs::Read(dataset+"/codes.bin.M"+to_string(PQ_M)+ "K"+to_string(PQ_K)+".sorted", &bitvecs);
            cout << "dataset read "<<get_current_time_str() << endl;
        } else {
            cout << "sorted database does not exist" << endl;
            BitVecs::Read(dataset+"/codes.bin.M"+to_string(PQ_M)+ "K"+to_string(PQ_K), &bitvecs);
            cout << "dataset read "<<get_current_time_str() << endl;
            bitvecs.SortVecs();
            cout << "dataset sorted "<<get_current_time_str() << endl;
            BitVecs::Write(dataset+"/codes.bin.M"+to_string(PQ_M)+ "K"+to_string(PQ_K)+".sorted", bitvecs);
            cout <<"sorted dataset written " <<get_current_time_str()<< endl;
        }

        
        // (4) Read the codes
            #ifdef QUERY_DEBUG
        cout << "Reading unsorted codes" << endl;
        cout << dataset << endl;
        BitVecs::Read(dataset, &bitvecs); 
        PQ pq(PQ::ReadCodewords(dataset + "/codewords.txt")); 
            #else   // QUERY_DEBUG
        PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+ "K"+to_string(PQ_K)+"relocated_codewords.txt"));
            #endif  // QUERY_DEBUG
        PQTree pqtree(pq.GetCodewords(), &bitvecs);
        #endif  // PLAIN_PQ

        // (5) Search
        // ranked_scores[q][k] : top-k th result of the q-th query.
        
        double t0 = Elapsed();

        if (query_size != -1)
        queries.resize(query_size);
        vector<vector<pair<int, float> > >
                ranked_scores(queries.size(), vector<pair<int, float> >(top_k));

        double vm, rss;
        process_mem_usage(vm, rss);
        cout << "   query processing approx VM: " << vm << " KB; RSS: " << rss <<" KB" <<endl;
        cout << "queries size is "<<queries.size() << endl;
        int interval = N/query_size;
        if (random_sample) {
            for(int q = 0; q < (int) queries.size(); ++q){
                //ranked_scores[q] =pqtree.Query(queries[q], top_k);
                ranked_scores[q] =pqtree.SampledQuery(queries[q], interval*q);
                if (debug == 1) {
                    cout << ranked_scores[q][0].first << " "
                        << ranked_scores[q][0].second << endl;
                }
            }
        } else {
//            #pragma omp parallel for
            for(int q = 0; q < (int) queries.size(); ++q){
                ranked_scores[q] =pqtree.Query(queries[q], top_k);
                if (debug == 1) {
                    cout << queries[q][0] << " "
                        << ranked_scores[q][0].first << " "
                        << ranked_scores[q][0].second << endl;
                }
            }

        }
        for (int i = 0; i <= PQ_M; i ++) {
            cout << pqtree.dist_hist[i] << endl;
        }
        cout << (Elapsed() - t0) / queries.size() * 1000 << " [msec/query] " << endl;
        cout <<get_current_time_str()<< endl;

        //// (5) Write scores
        //WriteScores("score.txt", ranked_scores);
    }
    if (task == "groundtruth") {
        // use a similar process to 
        cout << "Generating groundtruth" << endl;
        cout <<  get_current_time_str()  << endl;

        // Read dataset
        uchar* vecs;
        long N = NN;
        cout << "N = " << N << endl;
        // Read codewords
        //PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+"K"+to_string(PQ_K)+"codewords.txt"));
        if (query_size == -1 ) {
            cout << "Please specify number of queries to run: -query_size " << endl;
            return 0;
        }
        // Read query vectors
        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext, query_size);
//        for (int i = query_size-2; i < query_size; i ++) {
//            for (int j = 0; j < queries[i].size() ; j ++) {
//                cout << queries[i][j] << " ";
//            }
//            cout << endl;
//        }
        queries.resize(query_size);
        // read partial raw data and find top_k neighbors part by part
        ItrReader reader(dataset+"/base."+ext, ext);
        //ItrReader reader(dataset+"/total."+ext, ext);
        vector<vector<float> > buff;  // Buffer
        long long  buff_max_size = N / 2000;
        if (buff_max_size < 100000) buff_max_size = 100000;
        long long vectors_checked= 0;
        int debug = 0;
        
        double t0 = Elapsed();
        // prepare results (min heaps)
        vector<vector<pair<uint, float>>> results;   // <vector_id, distance>
        results.resize(query_size);
        for (int i = 0; i < query_size; i ++) {
            results[i].resize(top_k);
            for (int j = 0; j < top_k; j ++) {
                results[i][j].second = FLT_MAX;
            }
            make_heap(results[i].begin(), results[i].end(), comp_max_heap);
        }

        vector<priority_queue<pair<float, uint>, 
               vector<pair<float, uint>>, 
               decltype(cmp_max)>> max_heaps;
        for (int i = 0; i < query_size; i ++) {
            priority_queue<pair<float, uint>, vector<pair<float, uint>>,
                                decltype(cmp_max)> max_heap(cmp_max);
            max_heaps.push_back(max_heap);
        }
        // start to scanning
        cout << "Start scanning " <<get_current_time_str() << endl;
        cout << "max buffer size " << buff_max_size << endl;
        while(!reader.IsEnd()){
            buff.push_back(reader.Next());  // Read a line (a vector) into the buffer
            if( (int) buff.size() == buff_max_size){  // (1) If buff_max_size vectors are read,
                // batch topk queries
                //batch_partial_topk_queries(buff, queries, results, vectors_checked);
                batch_partial_topk_queries(buff, queries, max_heaps, top_k, vectors_checked);
                vectors_checked+= (int) buff.size();  // update id
                buff.clear(); // (3) refresh
                printf("\r%lld/%lld  %.1f%%     ",vectors_checked, N, ((double)vectors_checked)*100/N);
                fflush(stdout);
                if (vectors_checked/buff_max_size % 10 == 0)
                    cout << vectors_checked<< " / " << N << " vectors are checked in total" << endl;
            }
            if (vectors_checked== N) break;
        }
        cout << "debug " <<debug<< endl;
        if(0 < (int) buff.size()){ // Rest
            // batch topk queries
            //batch_partial_topk_queries(buff, queries, results, vectors_checked);
            batch_partial_topk_queries(buff, queries, max_heaps, top_k, vectors_checked);
            vectors_checked += (int) buff.size();
            buff.clear();
        }
        cout << vectors_checked << endl;
        N = vectors_checked;
        cout <<get_current_time_str() << endl;
        cout << (Elapsed() - t0) / queries.size() * 1000 << " [msec/query] " << endl;

        for (int i = 0; i < max_heaps.size(); i ++) {
            int index = top_k-1;
            while (max_heaps[i].size() != 0) {
                results[i][index].first = max_heaps[i].top().second;
                results[i][index].second = max_heaps[i].top().first;
                index --;
                max_heaps[i].pop();
            }
        }
        // write results
        string results_file = dataset + "/groundtruth/"+
                             + "N" + to_string(N)
                             + "Top" + to_string(top_k)
                             + ".txt";
        PQBase::write_groundtruth(results_file, PQ_M, PQ_K, results, query_size, top_k);
        cout << "Groundtruth written to " << results_file << endl;
    }
    if (task == "accuracy") {
        cout <<"Checking accuracy " <<get_current_time_str()<<endl;
        // (1) Make sure you've already trained codewords and encoded database vectors
        // (2) Read query vectors
        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        
        uchar* vecs;
        long N;
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+to_string(PQ_K), vecs, N);
        if (NN != -1) N = NN;
        cout << "N = " << N << endl;
        PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+"K"+to_string(PQ_K)+"codewords.txt"));
        PQTree pqtree(pq.GetCodewords(), vecs, N);

        // read ground truth
        vector<vector<pair<uint, float>>> results;   // <vector_id, distance>
        string results_file = dataset + "/groundtruth/"+
                             + "N" + to_string(N)
                             + "Top10000"
                             + ".txt";
        PQBase::read_groundtruth(results_file, PQ_M, PQ_K, results, query_size, top_k);
        // (5) Search
        // ranked_scores[q][k] : top-k th result of the q-th query.
        
        vector<vector<pair<int, float> > >
                ranked_scores(queries.size(), vector<pair<int, float> >(top_k));

        double t0 = Elapsed();
        double total_score = 0.0;
        if (query_size != -1)   queries.resize(query_size);
        cout << "queries size is "<<queries.size() << endl;
        vector<pair<uint, float>> dist_pairs;
        dist_pairs.resize(N);
        for(int q = 0; q < (int) queries.size(); ++q){

            pqtree.QueryPlain(queries[q], top_k, dist_pairs);
            nth_element(dist_pairs.begin(), dist_pairs.begin()+top_k, 
                        dist_pairs.end(),
                       [](const pair<uint, float>& a, const pair<uint, float>& b) {
                return a.second < b.second;
            });
            // compare to ground truth
            for (int i = 0; i < top_k; i ++) {
                for (int j = 0; j < top_k; j ++) {
                    if (ranked_scores[q][i].first == results[q][j].first)
                        total_score ++;
                }
            }
        }
        cout << total_score / (query_size*top_k) << endl;

        cout << (Elapsed() - t0) / queries.size() * 1000 << " [msec/query] " << endl;
        cout <<get_current_time_str()<< endl;

        //// (5) Write scores
        //WriteScores("score.txt", ranked_scores);
    }
    if (task == "recall") {
        cout <<"Checking recall " <<get_current_time_str()<<endl;
        // (1) Make sure you've already trained codewords and encoded database vectors
        // (2) Read query vectors
        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        
        uchar* vecs;
        long N;
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+
                                to_string(PQ_K), vecs, N);
        if (NN != -1) N = NN;
        cout << "N = " << N << endl;
        PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+"K"+
                                to_string(PQ_K)+"codewords.txt"));
        PQTree pqtree(pq.GetCodewords(), vecs, N);

        // read ground truth
        vector<vector<pair<uint, float>>> results;   // <vector_id, distance>
        string results_file = dataset + "/groundtruth/"+
                             + "N" + to_string(N)
                             + "Top10000" 
                             + ".txt";
        cout << results_file << endl;
        PQBase::read_groundtruth(results_file, PQ_M, PQ_K, results, query_size, top_k);
        // (5) Search
        // ranked_scores[q][k] : top-k th result of the q-th query.
        
        vector<vector<pair<int, float> > >
                ranked_scores(queries.size(), vector<pair<int, float> >(top_k));

        double t0 = Elapsed();
        double total_score = 0.0;
        if (query_size != -1)   queries.resize(query_size);
        cout << "queries size is "<<queries.size() << endl;
        vector<pair<uint, float>> dist_pairs;
        dist_pairs.resize(N);
        for(int q = 0; q < (int) queries.size(); ++q){

            pqtree.QueryPlain(queries[q], top_k, dist_pairs);
            nth_element(dist_pairs.begin(), dist_pairs.begin()+top_k, 
                        dist_pairs.end(),
                       [](const pair<uint, float>& a, const pair<uint, float>& b) {
                return a.second < b.second;
            });
            for (int i = 0; i < top_k; i ++) {
                ranked_scores[q][i].first = dist_pairs[i].first;
                ranked_scores[q][i].second = dist_pairs[i].second;
            }
            if (debug == 1) {
                cout << dist_pairs[0].first << " "
                    << dist_pairs[0].second << endl;
            }

            // compare to ground truth
            for (int i = 0; i < top_k; i ++) {
                for (int j = 0; j < top_k; j ++) {
                    if (ranked_scores[q][i].first == results[q][j].first)
                        total_score += 1;
                }
            }
            if (debug == 1) {
                cout << queries[q][0] << " "
                    << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
                cout << total_score/((q+1)*top_k) << endl;
            }
        }
        cout << total_score / (query_size*top_k) << endl;

        cout << (Elapsed() - t0) / queries.size() * 1000 << " [msec/query] " << endl;
        cout <<get_current_time_str()<< endl;

        //// (5) Write scores
        //WriteScores("score.txt", ranked_scores);
    }
    if (task == "mAP") {    // mean Averate Precision
        cout <<"Checking accuracy " <<get_current_time_str()<<endl;
        // (1) Make sure you've already trained codewords and encoded database vectors
        // (2) Read query vectors
        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        
        uchar* vecs;
        int dim = queries[0].size();
        long N;
        if (NN != -1) N = NN;
        cout << "N = " << N << endl;
        //PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+to_string(PQ_K), vecs, N);
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)+"K"+to_string(PQ_K)+"N"+to_string(N), vecs, N);
        cout << "N = " << N << endl;
        PQ pq(PQ::ReadCodewords(dataset + "/M"+to_string(PQ_M)+"K"+to_string(PQ_K)+"codewords.txt"));
        PQTree pqtree(pq.GetCodewords(), vecs, N);

        // read ground truth
        vector<vector<pair<uint, float>>> results;   // <vector_id, distance>
        string results_file = dataset + "/groundtruth/"+
                             + "N" + to_string(N)
                             + "Top" + to_string(top_k)
                             + ".txt";
        PQBase::read_groundtruth(results_file, PQ_M, PQ_K, results, query_size, top_k);
        // (5) Search
        // ranked_scores[q][k] : top-k th result of the q-th query.
        
        vector<vector<pair<int, float> > >
                ranked_scores(queries.size(), vector<pair<int, float> >(top_k));

        double t0 = Elapsed();
        double total_score = 0.0;
        double avg_ratio = 0.0;
        double max_ratio = 0.0;
        if (query_size != -1)   queries.resize(query_size);
        cout << "queries size is "<<queries.size() << endl;
        vector<pair<uint, float>> dist_pairs;
        dist_pairs.resize(N);
        for(int q = 0; q < (int) queries.size(); ++q){

            pqtree.QueryPlain(queries[q], top_k, dist_pairs);
            nth_element(dist_pairs.begin(), dist_pairs.begin()+top_k, 
                        dist_pairs.end(),
                       [](const pair<uint, float>& a, const pair<uint, float>& b) {
                return a.second < b.second;
            });
            if (debug == 1) {
                cout << dist_pairs[0].first << " "
                    << dist_pairs[0].second << endl;
            }

            ranked_scores[q] =pqtree.Query(queries[q], top_k);
            sort(ranked_scores[q].begin(), ranked_scores[q].end(), [](const pair<int,float>& a, 
                const pair<int, float>& b) {
                return a.second < b.second;
            });
            if (debug == 1) {
                cout << "Results "
                    << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
            // calculate mAP compare to ground truth
            if (debug)
                cout << "Groundtruth " << results[q][0].first << " " << results[q][0].second << endl;
            double apk = 0; // average precision @ k
            for (int k = 1; k <= top_k; k ++) {
                double score = 0;
                double ratio = sqrt(results[q][k-1].second) / 
                                sqrt(ranked_scores[q][k-1].second);
                if (ratio > max_ratio) max_ratio = ratio;
                avg_ratio += ratio;
                for (int i = 0; i < k; i ++) {
                    for (int j = 0; j < k; j ++) {
                        if (ranked_scores[q][i].first == results[q][j].first)
                            score ++;
                    }
                }
                apk += score/k;
            }
            total_score += apk/top_k;
        }
        // write results to file
        string DPQ_results_file = dataset + "/DeltaPQResults/"+
                             + "N" + to_string(N)
                             + "Top" + to_string(top_k)
                             + ".txt";
        PQBase::write_groundtruth(DPQ_results_file, PQ_M, PQ_K, results, query_size, top_k);
        cout << "Results written to " << results_file << endl;

        cout << "MAP = " << total_score / (query_size) << endl;
        cout << "Avg ratio: " << avg_ratio / (top_k * queries.size()) << endl;
        cout << "Max ratio: " << max_ratio << endl;
        cout << (Elapsed() - t0) / queries.size() * 1000 << " [msec/query] " << endl;
        cout <<get_current_time_str()<< endl;
        string file_name = dataset+"/base."+ext;
        fstream myFile(file_name, ios::in | ios::binary);
        double avg_recall_eps = 0;
        double avg_recall = 0;
        double avg_k_app_ratio=0;
        long long bvec_len = 4 + dim;
        long long fvec_len = 4 + 4 * dim;
        float* fvec = new float[dim];
        uchar* bvec = new uchar[dim];
        for (int q = 0; q < queries.size(); q ++) {
            double kth = sqrt(results[q][top_k-1].second);
            double thres = sqrt(results[q][top_k-1].second) * 1.1;
            double max = 0;
            for (int i = 0; i < top_k; i ++) {
                // get each vector;
                long long idx = ranked_scores[q][i].first;
                double distance = 0;
                if (ext == "bvecs") {
                    myFile.seekg(idx*bvec_len + 4);   // omit dimension info
                    myFile.read((char*)bvec, dim);             // read uchars
                    for (int d = 0; d < dim; d ++) {
                        distance += pow(queries[q][d]-bvec[d], 2);
                    }
                } else {
                    myFile.seekg(idx*fvec_len + 4);
                    myFile.read((char*)fvec, dim*4);           // read floats
                    for (int d = 0; d < dim; d ++) {
                        distance += pow(queries[q][d]-fvec[d], 2);
                    }
                }
                distance = sqrt(distance);
                if (distance <= thres) avg_recall_eps += 1;
                if (distance <= kth) avg_recall += 1;
                if (distance > max) max = distance;
            }
            if (kth == 0)  {
                cout << "kth = 0" << endl;
            }else avg_k_app_ratio += max / kth;
        }
        cout << "eps recall = " << avg_recall_eps / (top_k * queries.size()) << endl;
        cout << "recall = " << avg_recall / (top_k * queries.size()) << endl;
        cout << "ratio = " << avg_k_app_ratio / ( queries.size()) << endl;
        cout <<get_current_time_str()<< endl;

        //// (5) Write scores
        //WriteScores("score.txt", ranked_scores);
    }
    return 0;
}
