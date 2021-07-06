#ifndef PQTREE_H
#define PQTREE_H
// PQTree
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <bitset>
#include <limits>

#include "pq.h"
using namespace std;

#define CL 64

#define PLAIN_PQ
//#define ENCODE_DEBUG 1
//#define QUERY_DEBUG 1
#define BOUNDS_TABLE 1
//#define ASSIGN_MODULE 1
#define SORTED_BOUND 1

extern int PQ_M;
extern int PQ_K;
extern int with_id;

class BitVecs{
public:
    BitVecs() : m_N(0), m_D(0) {}   // Space is not allocated. You need to call Resize first
    BitVecs(int N, int D) { Resize(N, D); }  // Space is allocated
    BitVecs(const vector<bitset<CL> > &vec);  // vec<vec<bool>> -> BitVecs


    void Resize(int N, int D); // After resized, the old values are remained

    // Getter
    const bool GetVal(int n, int d) const;     // n-th vec, d-th dim
    bitset<CL> GetVec(int n) const;     // n-th vec

    // Setter
    void SetVal(int n, int d, bool val);
    void SetVec(int n, const bitset<CL> vec);

    // IO
    static void Write(string path, const BitVecs &vecs);
    static void Read(string path, BitVecs *vecs, int top_n = -1); // Read top_n codes. if top_n==-1, read all
    static BitVecs Read(string path, int top_n = -1); // wrapper.

    // Be careful
    const bool *RawDataPtr() const;

    int Size() const {return m_N;}
    int Dim() const {return m_D;}

    void SortVecs();
private:
    int m_N;
    int m_D;
    vector<bitset<CL>> m_data; // a long array

};


class PQTree {
    
    class Node {
    public:
        static const int max_level=8; // 2^8 = 256 centroids
        int level;
        int size;
        bool isLeaf=false;
        Node** children;
        float* upper_bounds;
        float* lower_bounds;
        bitset<CL> prefix;
        Node();
        ~Node();
        Node(int level, bitset<CL> prfx);
        void CalBounds(const vector<vector<float>>& sub_distances);
    };

    // need codewords here
    // a codewords clustering function (recursive dichotomy)
private:
    vector<PQ::Array> m_codewords; // [ns][ks][ds], original codewords
    vector<PQ::Array> m_codewords_t; // [ns][ks][ds], reordered codewords
                                     // by clustering
    int m_M;
    int m_Ks;
    int m_Ds;

    Node* root;
    float** m_sub_distances; // M x K, will change per query
    float** m_sub_bounds;   // upper/lower bound table (changes per query)
public:
    float EPS = 0.000001;
    BitVecs* database;
    #ifdef PLAIN_PQ
    vector<int> dist_hist;
    uchar* plain_database;
    PQTree(std::vector<PQ::Array> codewords, uchar* vecs, long length);
    vector<pair<int, float> > QueryPlain(const vector<float> &query, int top_k); // for top-k search
    #endif
    void QueryPlain(const vector<float> &query, int top_k, vector<pair<uint, float>>& dist_pairs); // for top-k search
    long N;
    int n_Nodes;
    int n_vecs=0; // DEBUG
    PQTree(std::vector<PQ::Array> codewords);
    PQTree(std::vector<PQ::Array> codewords, BitVecs* bitvecs);
    void DichotomizeCodewords(int start, int end);
    vector<PQ::Array> GetCodewordsT();
    bitset<CL> Encode(const vector<float> &vec) const;
    vector<uchar> EncodePlain(vector<float> &vec);
    BitVecs Encode(const vector<vector<float> > &vecs) const;
    uchar* EncodePlain(vector<vector<float> > &vecs);
    vector<pair<int, float> > Query(const vector<float> &query, int top_k); // for top-k search
    vector<pair<int, float> > SampledQuery(vector<float> &query, int id); // for top-k search
    long search_prefix(unsigned long prefix, long start, long end);
    void BuildTree(Node* root, float& m_upper_bound);
    void BuildTreeSorted(Node* root, float& m_upper_bound);
    inline int GetM(){return m_M;}
    inline float CalVectorDist(const vector<float> & a,const vector<float> &b);
    inline float CalCodeDist(const bitset<CL> code);
    vector<float> Decode(const bitset<CL> code);
    vector<float> DecodePlain(const uchar* code);
    float MinDist(const vector<float> qeury);
    void ScanLeaves(Node* root, const float bound, float& dist, int& id);
    // IO function
    static void Write(string path, const uchar* vecs, long N);
    static void Read(string path, uchar*& vecs, long& n, int top_n= -1);

    float CalCoarseBound(const bitset<CL> prefix);
};
#endif
