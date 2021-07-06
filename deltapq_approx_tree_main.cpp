
#include "pq_tree.h"
#include "utils.h"
#include "deltapq_create_approx_tree.h"

using namespace std;

int PQ_M;
int PQ_K;
int with_id=0;
string ext = "fvecs";
int dim=128;

int main(int argc, char* argv[]){
    string dataset;
    string queryset;
    string grndtruth;
    int query_size=-1;
    string task = "approx_tree";
    int top_k = 1;
    long long N =-1;
    int diff_argument = 1;
    int debug = 0;
    int max_height_folds = 1;
    int method=1;   // 1: tree wh 2: tree w/o h 3: clique 4: block-aware design
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") {
    		dataset = string(argv[i + 1]);
        }
        if (arg == "-queryset") {
    		queryset = string(argv[i + 1]);
        }
        if (arg == "-task") {
            task = string(argv[i + 1]);
        }
        if (arg == "-topk") {
            top_k = atoi(argv[i + 1]);
        }
        if (arg == "-N") {
            N = atoll(argv[i+1]);
        }
        if (arg == "-diff") {
            diff_argument = atoi(argv[i+1]);
        }
        if (arg == "-query_size") {
            query_size = atoi(argv[i+1]);
        }
        if (arg == "-m") {
            PQ_M = atoi(argv[i+1]);
        }
        if (arg == "-k") {
            PQ_K = atoi(argv[i+1]);
        }
        if (arg == "-ext") {
    		ext = string(argv[i + 1]);
        }
        if (arg == "-debug") {
            debug = 1;
        }
        if (arg == "-h") {
            max_height_folds = atoi(argv[i+1]);
        }
        if (arg == "approx_with_id") {
            with_id = 1;
        }
        if (arg == "-method") {
            method = atoi(argv[i+1]);
        }
    }

    if (task == "approx_tree") {
        cout << "M = " << PQ_M << endl;
        uchar* vecs;
        long NN; // NN will be read from file
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)
                        +"K"+to_string(PQ_K)+"N"+to_string(N), vecs, NN);
        if (N == -1) {
            N = NN;
        }
        for (long long i = 0; i < 10; i ++) {
            for (int m = 0; m < PQ_M; m ++) {
                cout << (int)vecs[i*PQ_M+m] << " ";
            }
            cout << endl;
        }
        
        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"building graph " << get_current_time_str()<<endl;

        // start to query or index
        uint num_codes = N;
        double t0 = Elapsed();
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;
        {
            dist_tables = new float*[PQ_M];
            for (int m = 0; m < PQ_M; m ++) {
                dist_tables[m] = new float[PQ_K*PQ_K];
            }
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
        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        //
        // start to query or index
        QNode* nodes;
//        nodes = new QNode[N+1];
//        memset(nodes, 0, sizeof(QNode)*(N+1));
        diff_argument = PQ_M;
        m_sub_distances = new float*[PQ_M]; // m_sub_distances defined in .h file
        for (int i = 0; i < PQ_M; i++) {
            m_sub_distances[i] = new float[PQ_K];
            memset(m_sub_distances[i], 0, sizeof(float)*PQ_K);
        }
        bool file_read = create_approx_tree(dataset, vecs, dummycodes, 
                        dummymarks, PQ_M, PQ_K, num_codes, diff_argument, 
                        nodes, m_sub_distances,
                        dist_tables, max_height_folds, method);
        cout << "==========================BUILD DELTATREE INDEX IN " << (Elapsed()-t0) <<" [sec] "
             << "==========================" << endl << endl;;
        if (!file_read)
        {
            cout << "WARNING: Just built an index. no query processed." << endl;
            return 0;
        }
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"build graph done" << get_current_time_str()<<endl;

    }
    if (task == "diff_index") {
        // find diffs of adjacent pqcodes
        cout << "M = " << PQ_M << endl;
        uchar* vecs;
        long NN; // NN will be read from file
        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)
                        +"K"+to_string(PQ_K), vecs, NN);
        if (N == -1) {
            N = NN;
        }
        // INFO
        for (long long i = 0; i < 10; i ++) {
            for (int m = 0; m < PQ_M; m ++) {
                cout << (int)vecs[i*PQ_M+m] << " ";
            }
            cout << endl;
        }
        // END OF INFO
        
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"building graph " << get_current_time_str()<<endl;

        // start to query or index
        uint num_codes = N;
        
        create_diff_index(dataset, vecs, PQ_M, PQ_K, num_codes);
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"build graph done" << get_current_time_str()<<endl;

    }
    if (task == "approx_with_id") {
        cout << "M = " << PQ_M << endl;
        with_id = 1;
        uchar* vecs;
        long NN=-1; // NN will be read from file
        PQTree::Read(dataset+"/codes_with_ids.bin.plain.M"+to_string(PQ_M)
                        +"K"+to_string(PQ_K), vecs, NN);
        if (N == -1) {
            N = NN;
        }
        // INFO
        PQ_M += sizeof(int);
        for (long long i = 0; i < 10; i ++) {
            for (int m = 0; m < PQ_M; m ++) {
                cout << (int)vecs[i*PQ_M+m] << " ";
            }
            cout << endl;
        }
        PQ_M -= sizeof(int);
        // INFO

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"building graph " << get_current_time_str()<<endl;

        // start to query or index
        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;
        {
            dist_tables = new float*[PQ_M];
            for (int m = 0; m < PQ_M; m ++) {
                dist_tables[m] = new float[PQ_K*PQ_K];
            }
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
        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        //
        // start to query or index
        QNode* nodes=NULL;
//        nodes = new QNode[N+1];
//        memset(nodes, 0, sizeof(QNode)*(N+1));

        int M_arg;
        M_arg = PQ_M + sizeof(int);
        diff_argument = PQ_M + sizeof(int);
        bool file_read = create_approx_tree(dataset, vecs, dummycodes, 
                        dummymarks, M_arg, PQ_K, num_codes, diff_argument, 
                        nodes, m_sub_distances,
                        dist_tables, max_height_folds);
        if (!file_read)
        {
            cout << "Built approx tree index." << endl;
            return 0;
        }
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;
        cout <<"Approx Tree W H done" << get_current_time_str()<<endl;

    }
    if (task == "query") {
        long NN=-1; // NN will be read from file
        uchar* vecs=NULL;
//        PQTree::Read(dataset+"/codes.bin.plain.M"+to_string(PQ_M)
//                        +"K"+to_string(PQ_K), vecs, NN);
        if (N == -1) {
            N = NN;
        }

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        // start to query or index
        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        

        int M_arg;
        if (with_id) {
            M_arg = PQ_M + sizeof(int);
            diff_argument = PQ_M + sizeof(int);
        } else M_arg = PQ_M;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        cout << queries.size() << " query vectors read from " 
             << dataset + "/query." + ext << endl;
        if (queries.size() > 10000) queries.resize(10000);
        if (query_size != -1)
            queries.resize(query_size);

        vector<vector<pair<int, float>>> ranked_scores(queries.size(), 
                                        vector<pair<int,float> >(top_k));
        uchar** decoder = new uchar*[256];
        uchar masks[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        for (int i = 0; i < 256; i++) {
            vector<uchar> slots;
            uchar bits = (uchar)i;
            for (uchar j = 0; j < 8; j++)
                if (masks[j] == (bits & masks[j]))
                    slots.push_back(j);
        
            decoder[i] = new uchar[slots.size() + 1]; 
            decoder[i][0] = (uchar)slots.size();
            for (uchar j = 0; j < slots.size(); j++)
                decoder[i][j + 1] = (uchar)slots[j];
        }
        
        double t0 = Elapsed();
        for(int q = 0; q < (int) queries.size(); ++q) {
            // row store query
            //row_store_query_processing_scan_compressed_codes_opt_o_direct(dataset, queries[q], top_k,
            // direct I/O from disk
            query_processing_scan_compressed_codes_opt_o_direct(dataset, queries[q], top_k,
            // other methods
            //query_processing_scan_compressed_codes_opt(dataset, queries[q], top_k,
            //query_processing_scan_compressed_codes(dataset, queries[q], top_k,
                            PQ_M, PQ_K, pq.GetDs(),
                            num_codes, 
                            pq.GetCodewords(), 
                            ranked_scores[q], decoder);
            if (debug) {
                cout << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
        }
        cout << (Elapsed()-t0) / queries.size()*1000 <<" [msec/query] "<< endl;
        cout <<get_current_time_str()<< endl;
        cout << queries.size() << " queries run" << endl;

    }
    if (task == "batch_query") {
        long NN=-1; // NN will be read from file
        if (N == -1) {
            N = NN;
        }

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        // start to query or index
        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        

        int M_arg;
        if (with_id) {
            M_arg = PQ_M + sizeof(int);
            diff_argument = PQ_M + sizeof(int);
        } else M_arg = PQ_M;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        cout << queries.size() << " query vectors read from " 
             << dataset + "/query." + ext << endl;
        if (queries.size() > 10000) queries.resize(10000);
        if (query_size != -1)
            queries.resize(query_size);

        vector<vector<pair<int, float>>> ranked_scores(queries.size(), 
                                        vector<pair<int,float> >(top_k));
        uchar** decoder = new uchar*[256];
        uchar masks[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        for (int i = 0; i < 256; i++) {
            vector<uchar> slots;
            uchar bits = (uchar)i;
            for (uchar j = 0; j < 8; j++)
                if (masks[j] == (bits & masks[j]))
                    slots.push_back(j);
        
            decoder[i] = new uchar[slots.size() + 1]; 
            decoder[i][0] = (uchar)slots.size();
            for (uchar j = 0; j < slots.size(); j++)
                decoder[i][j + 1] = (uchar)slots[j];
        }
        
        double t0 = Elapsed();
        // direct I/O from disk
        //query_processing_batch_scan_compressed_codes_opt_o_direct(dataset, queries, top_k,
        query_processing_opt_batch_scan_compressed_codes_opt_o_direct(dataset, queries, top_k,
                        PQ_M, PQ_K, pq.GetDs(),
                        num_codes, 
                        pq.GetCodewords(), 
                        ranked_scores, decoder);
        for(int q = 0; q < (int) queries.size(); ++q) {
            if (debug) {
                cout << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
        }
        cout << (Elapsed()-t0) / queries.size()*1000 <<" [msec/query] "<< endl;
        cout << get_current_time_str()<< endl;
        cout << queries.size() << " queries run" << endl;

    }
    if (task == "diff_scan") {
        long NN; // NN will be read from file

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << "top_k = " << top_k << endl;
        cout << dataset << endl;

        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        uchar** decoder = new uchar*[256];
        uchar masks[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        for (int i = 0; i < 256; i++) {
            vector<uchar> slots;
            uchar bits = (uchar)i;
            for (uchar j = 0; j < 8; j++)
                if (masks[j] == (bits & masks[j]))
                    slots.push_back(j);
        
            decoder[i] = new uchar[slots.size() + 1]; 
            decoder[i][0] = (uchar)slots.size();
            for (uchar j = 0; j < slots.size(); j++)
                decoder[i][j + 1] = (uchar)slots[j];
        }

        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        cout << queries.size() << " query vectors read from " 
             << dataset + "/query." + ext << endl;
        if (queries.size() > 10000) queries.resize(10000);
        if (query_size != -1)
            queries.resize(query_size);

        vector<vector<pair<int, float>>> ranked_scores(queries.size(), 
                                        vector<pair<int,float> >(top_k));
        
        double t0 = Elapsed();
        for(int q = 0; q < (int) queries.size(); ++q) {
            query_processing_diff_scan_o_direct(
                            dataset, queries[q], top_k,
                            PQ_M, PQ_K, pq.GetDs(),
                            num_codes, 
                            pq.GetCodewords(), 
                            ranked_scores[q], decoder);
            if (debug) {
                cout << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
        }
        cout << (Elapsed()-t0) / queries.size()*1000 <<" [msec/query] "<< endl;
        cout <<get_current_time_str()<< endl;
        cout << queries.size() << " queries run" << endl;

    }
    if (task == "pqscan") {
        long NN; // NN will be read from file

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << "top_k = " << top_k << endl;
        cout << dataset << endl;

        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        

        int M_arg;
        if (with_id) {
            M_arg = PQ_M + sizeof(int);
            diff_argument = PQ_M + sizeof(int);
        } else M_arg = PQ_M;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        cout << queries.size() << " query vectors read from " 
             << dataset + "/query." + ext << endl;
        if (queries.size() > 10000) queries.resize(10000);
        if (query_size != -1)
            queries.resize(query_size);

        vector<vector<pair<int, float>>> ranked_scores(queries.size(), 
                                        vector<pair<int,float> >(top_k));
        
        double t0 = Elapsed();
        for(int q = 0; q < (int) queries.size(); ++q) {
            //query_processing_scan_pqcodes_o_direct(
            query_processing_scan_pqcodes_o_direct(
                            dataset, queries[q], top_k,
                            PQ_M, PQ_K, pq.GetDs(),
                            num_codes, 
                            pq.GetCodewords(), 
                            ranked_scores[q]);
            if (debug) {
                cout << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
        }
        cout << (Elapsed()-t0) / queries.size()*1000 <<" [msec/query] "<< endl;
        cout <<get_current_time_str()<< endl;
        cout << queries.size() << " queries run" << endl;

    }
    if (task == "update") { // simulate update
        long NN=-1; // NN will be read from file
        if (N == -1) {
            N = NN;
        }

        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        // start to query or index
        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        

        int M_arg;
        if (with_id) {
            M_arg = PQ_M + sizeof(int);
            diff_argument = PQ_M + sizeof(int);
        } else M_arg = PQ_M;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        uchar** decoder = new uchar*[256];
        uchar masks[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        for (int i = 0; i < 256; i++) {
            vector<uchar> slots;
            uchar bits = (uchar)i;
            for (uchar j = 0; j < 8; j++)
                if (masks[j] == (bits & masks[j]))
                    slots.push_back(j);
        
            decoder[i] = new uchar[slots.size() + 1]; 
            decoder[i][0] = (uchar)slots.size();
            for (uchar j = 0; j < slots.size(); j++)
                decoder[i][j + 1] = (uchar)slots[j];
        }
        
        double t0 = Elapsed();
        update_query_processing_simulation(dataset, 
                        PQ_M, PQ_K, pq.GetDs(),
                        num_codes, query_size, 
                        pq.GetCodewords(), 
                        decoder);
        cout << (Elapsed()-t0)/query_size*1000 <<" [msec] "<< endl;
        cout <<get_current_time_str()<< endl;

    }
    if (task == "query_im") {   // in-memory query

        // read compressed code from disk
        string file_name = dataset + "/M" + to_string(PQ_M) + "K" 
                        + to_string(PQ_K) + "_Approx_compressed_codes_opt";
        file_name = file_name + "_N" + to_string(N);
        cout << file_name << endl;
        ifstream ifs(file_name, ios::binary);
        long long n_codes;
        long long n_bytes;
        ifs.read((char*) &(n_codes), sizeof(long long));
        ifs.read((char*) &(n_bytes), sizeof(long long));
        if (N != n_codes) {
            cout << "scan only part of the codes " << N << " / "
                << n_codes << endl;
        }
        uchar* codes = new uchar[n_bytes];
        ifs.read((char*) codes, sizeof(uchar) * (n_bytes));
        for (int i = 0; i < PQ_M; i ++) {
            cout << (int) codes[i] << " ";
        }
        cout << endl;
        PQ pq(PQ::ReadCodewords(dataset +"/M"+to_string(PQ_M)+"K"
                                +to_string(PQ_K)+"codewords.txt"));
        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        // start to query or index
        uint num_codes = N;
        
        const vector<vector<vector<float>>>& m_codewords = pq.GetCodewords();
        int m_Ds = pq.GetDs();
        float** dist_tables;

        vector<uchar> dummycodes;
        vector<bool> dummymarks;
        

        int M_arg;
        if (with_id) {
            M_arg = PQ_M + sizeof(int);
            diff_argument = PQ_M + sizeof(int);
        } else M_arg = PQ_M;

        cout << "M = " << PQ_M << endl;
        cout << "K = " << PQ_K << endl;
        cout << "N = " << N << endl;
        cout << dataset << endl;

        vector<vector<float> > queries = ReadTopN(dataset + "/query." + ext, ext);
        cout << queries.size() << " query vectors read from " 
             << dataset + "/query." + ext << endl;
        if (queries.size() > 10000) queries.resize(10000);
        if (query_size != -1)
            queries.resize(query_size);

        vector<vector<pair<int, float>>> ranked_scores(queries.size(), 
                                        vector<pair<int,float> >(top_k));
        uchar** decoder = new uchar*[256];
        uchar masks[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        for (int i = 0; i < 256; i++) {
            vector<uchar> slots;
            uchar bits = (uchar)i;
            for (uchar j = 0; j < 8; j++)
                if (masks[j] == (bits & masks[j]))
                    slots.push_back(j);
        
            decoder[i] = new uchar[slots.size() + 1]; 
            decoder[i][0] = (uchar)slots.size();
            for (uchar j = 0; j < slots.size(); j++)
                decoder[i][j + 1] = (uchar)slots[j];
        }
        
        double t0 = Elapsed();
        for(int q = 0; q < (int) queries.size(); ++q) {
            // in memory query on compressed codes 
            query_processing_scan_compressed_codes_opt_in_memory(codes,n_bytes,
                            queries[q], top_k,
                            PQ_M, PQ_K, pq.GetDs(),
                            num_codes, 
                            pq.GetCodewords(), 
                            ranked_scores[q], decoder);
            if (debug) {
                cout << ranked_scores[q][0].first << " "
                    << ranked_scores[q][0].second << endl;
            }
        }
        cout << (Elapsed()-t0) / queries.size()*1000 <<" [msec/query] "<< endl;
        cout <<get_current_time_str()<< endl;
        cout << queries.size() << " queries run" << endl;

    }
    cout << "===========================" << endl << endl;
    return 0;
}
