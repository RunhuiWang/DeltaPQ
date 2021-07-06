#include "utils.h"

//namespace pqtable {

FvecsItrReader::FvecsItrReader(std::string filename)
{
    ifs.open(filename, std::ios::binary);
    assert(ifs.is_open());
    Next();
}

bool FvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> FvecsItrReader::Next()
{
    std::vector<float> prev_vec = vec; // return the currently stored vec
    int D;
    if(ifs.read( (char *) &D, sizeof(int) )){ // read "D"
        // Then, read a D-dim vec
//        if (D <=0 || D > 1024) {
//            cout << "Dad data: D is " << D << endl;
//            exit(0);
//        }
        vec.resize(D); // allocate D-dim
        assert(ifs.read( (char *) vec.data(), sizeof(float) * D)); // Read D * float.
        eof_flag = false;
    }else{
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

BvecsItrReader::BvecsItrReader(std::string filename)
{
    ifs.open(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << filename << std::endl;
    }
    assert(ifs.is_open());
    Next();
}

bool BvecsItrReader::IsEnd() { return eof_flag; }

std::vector<float> BvecsItrReader::Next()
{
    std::vector<float> prev_vec = vec; // return the currently stored vec
    int D;
    if(ifs.read( (char *) &D, sizeof(int) )){ // read "D"
        //if (D != 128) {
        //    cout << D << endl;
        //    exit(0);
        //}
        // Then, read a D-dim vec
        vec.resize(D); // allocate D-dim
        std::vector<uchar> buff(D);
        assert(ifs.read( (char *) buff.data(), sizeof(uchar) * D)); // Read D * uchar.

        // Convert uchar to float
        for(int d = 0; d < D; ++d){
            vec[d] = static_cast<float>(buff[d]);
        }

        eof_flag = false;
    }else{
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

ItrReader::ItrReader(std::string filename, std::string ext){
    if(ext == "fvecs"){
        m_reader = (I_ItrReader *) new FvecsItrReader(filename);
    }else if(ext == "bvecs"){
        m_reader = (I_ItrReader *) new BvecsItrReader(filename);
    }else{
        std::cerr << "Error: strange ext type: " << ext << "in ItrReader" << std::endl;
        exit(1);
    }
}

ItrReader::~ItrReader(){
    delete m_reader;
}

bool ItrReader::IsEnd(){
    return m_reader->IsEnd();
}

std::vector<float> ItrReader::Next(){
    return m_reader->Next();
}

std::vector<std::vector<float> > ReadTopN(std::string filename, std::string ext, int top_n) {
    cout << "In ReadTopN " << filename << endl;
    std::vector<std::vector<float> > vecs;
    if(top_n != -1){
        vecs.reserve(top_n);
    }
    ItrReader reader(filename, ext);
    while(!reader.IsEnd()){
        if(top_n != -1 && top_n <= (int) vecs.size()){
            break;
        }
        vecs.push_back(reader.Next());
    }
    return vecs;
}

double Elapsed() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void WriteScores(std::string path, const std::vector<std::vector<std::pair<int, float> > > &scores)
{
    assert(!scores.empty());

    std::ofstream ofs(path);
    if(!ofs.is_open()){
        std::cerr << "Error: cannot open " << path << " in WriteScores" << std::endl;
        exit(1);
    }

    int query_sz = (int) scores.size();
    int top_k = (int) scores[0].size();

    ofs << query_sz << std::endl << top_k << std::endl;
    for(int q = 0; q < query_sz; ++q){
        for(int k = 0; k < top_k; ++k){
            ofs << scores[q][k].first << "," << scores[q][k].second << ",";
        }
        ofs << std::endl;
    }
}

// spinlock
void spinlock(atomic_flag& lock) {
    while (lock.test_and_set(std::memory_order_acquire)) {
        
    }
}

void unlock(atomic_flag& lock) {
    lock.clear(std::memory_order_release);
}

uint find_set(uint* parents, uint x) {
    if (x != parents[x]) {
        parents[x] = find_set(parents, parents[x]);
    }
    return parents[x];
}

uint find_set_read_only(uint* parents, uint x) {
    while (x != parents[x]) {
        x = parents[x];
    }
    return x;
}

void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

std::string get_current_time_str() {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
    std::string str(buffer);

    return str;
}
bool exists_test3 (const std::string& name) {
    
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}
void k_means_wrh(vector<vector<float>>& vecs, int k, int n_iter, 
            vector<vector<float>>& centers, vector<vector<int>>& label) {

}
