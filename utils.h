#ifndef PQTABLE_UTILS_H
#define PQTABLE_UTILS_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>
#include <atomic>
#include <unistd.h>
#include <iostream>
#include <sys/stat.h>   // for testing if file exists

#define LeafLastNode 1

// #define USE_PARENT_ID        // NodeP
using namespace std;
typedef unsigned int uint;
//typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
//typedef uint64_t uint128_t;

//namespace pqtable {

// Iterative reader class for reading .bvecs or .fvecs files.
// The next vector (std::vector<float>) is read by Next() function.
//
// Usage:
//   ItrReader reader("data.fvecs", "fvecs");
//   while(!reader.IsEnd()){
//     std::vector<float> vec = reader.Next();
//     /* some stuff for vec */
//   }
//
// Optional wrapper interface:
//   int top_n = 100;
//   std::vector<std::vector<float> > vecs = ReadTopN("data.fvecs", "fvecs", top_n);

// Interface (abstract basic class) of iterative reader
class I_ItrReader{
public:
    virtual ~I_ItrReader() {}
    virtual bool IsEnd() = 0;
    virtual std::vector<float> Next() = 0;
};

// Iterative reader for fvec file
class FvecsItrReader : I_ItrReader{
public:
    FvecsItrReader(std::string filename);
    bool IsEnd();
    std::vector<float> Next();
private:
    FvecsItrReader(); // prohibit default construct
    std::ifstream ifs;
    std::vector<float> vec; // store the next vec
    bool eof_flag;
    long long length;
};

// Iterative reader for bvec file
class BvecsItrReader : I_ItrReader{
public:
    BvecsItrReader(std::string filename);
    bool IsEnd();
    std::vector<float> Next(); // Read bvec, but return vec<float>
private:
    BvecsItrReader(); // prohibit default construct
    std::ifstream ifs;
    std::vector<float> vec; // store the next vec
    bool eof_flag;
};

// Proxy class
class ItrReader{
public:
    // ext must be "fvecs" or "bvecs"
    ItrReader(std::string filename, std::string ext);
    ~ItrReader();

    bool IsEnd();
    std::vector<float> Next();

private:
    ItrReader();
    I_ItrReader *m_reader;
};

// Wrapper. Read top-N vectors
// If top_n = -1, then read all vectors
std::vector<std::vector<float> > ReadTopN(std::string filename, std::string ext, int top_n = -1);






// Timer function
// Usage:
//   double t0 = Elapsed()
//   /* do something */
//   std::cout << Elapsed() - t0 << " [sec]" << std::endl;

double Elapsed();




// Output scores
// scores[q][k]: k-th score of q-th query, where each score is pair<int, float>.
// score[q][k].first: id,   score[q][k].second: distance
void WriteScores(std::string path,
                 const std::vector<std::vector<std::pair<int, float> > > &scores);


//}
bool exists_test3 (const std::string& name);

void spinlock(atomic_flag& lock);

void unlock(atomic_flag& lock);

uint find_set(uint* parents, uint x);

uint find_set_read_only(uint* parents, uint x);

void process_mem_usage(double& vm_usage, double& resident_set);
std::string get_current_time_str();
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);

        return h1 * 9967 +  h2;
    }
};

void k_means_wrh(vector<vector<float>>& vecs, int k, int n_iter, 
            vector<vector<float>>& centers, vector<vector<int>>& label);


#endif // PQTABLE_UTILS_H
