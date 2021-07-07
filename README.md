# DeltaPQ
DeltaPQ: Lossless Product Quantization Code Compression for High Dimensional Similarity Search

Paper link : http://www.vldb.org/pvldb/vol13/p3603-wang.pdf

## How to compile

### Prerequisite

1. OpenCV

2. Boost (required components: timer chrono system program_options)

### To compile
```
cmake .
make pqtree
make deltapq
```
## Usage

### Data preparation
The testing dataset can be obtained from http://corpus-texmex.irisa.fr

Under the folder of each dataset, rename the following files:
1. *_ base.*vecs -> base.*vecs
2. *_ learn.*vecs -> learn.*vecs
3. *_ query.*vecs -> query.*vecs

For example, after downloading and unzip siftsmall.tar.gz, rename siftsmall_base.fvecs to base.fvecs.

### Generate PQ codes
1. Learn codewords
```
./pqtree  -dataset [path to the dataset folder] 
          -ext     [file type]          // specify the input file type, either fvecs or bvecs
          -task learn                   // learn codewords for PQ
          -m       [M]                  // the number of sub-dimensions in PQ
          -k       [K]                  // the number of centroids in each sub-dimension in PQ 
          -train_size [training size]   // the number of vectors used for learning codewords
```
Example
```
./pqtree -dataset /data/local/pqdata/sift/ -ext fvecs -task learn -m 8 -k 256 - train_size 10000
```
2. Encode
```
./pqtree  -dataset [path to the dataset folder] 
          -ext     [file type]          // specify the input file type, either fvecs or bvecs
          -task encode                  // encode PQ codes
          -m       [M]                  // the number of sub-dimensions in PQ
          -k       [K]                  // the number of centroids in each sub-dimension in PQ 
          -N       [number of vectors]  // specify the number of vectors to use
```
Example
```
./pqtree -dataset /data/local/pqdata/sift/ -fvecs -task encode -m 8 -k 256
```

### Generate Approximate DeltaTree
```
./deltapq -dataset [path to the dataset folder] 
          -ext     [file type]          // specify the input file type, either fvecs or bvecs 
          -task approx_tree             // generate approximate DeltaTree
          -m       [M]                  // the number of sub-dimensions in PQ
          -k       [K]                  // the number of centroids in each sub-dimension in PQ 
          -h       [H]                  // controls the maximum tree height, usually set as 1
          -diff    [maximum weight]     // the maximum weight of edges in the tree, usually set as the same as m
          -N       [number of vectors]  // specify the number of vectors to use
```
Example
```
./deltapq -dataset /data/local/pqdata/sift/ -ext fvecs -task approx_tree -m 8 -k 256 -h 1 -diff 8 -N 1000000
```

### Query
```
./deltapq -dataset    [path to the dataset folder] 
          -ext        [file type]          // specify the input file type, either fvecs or bvecs 
          -task query                      // perform similarity queries on the deltatree
          -m          [M]                  // the number of sub-dimensions in PQ
          -k          [K]                  // the number of centroids in each sub-dimension in PQ 
          -h          [H]                  // controls the maximum tree height, usually set as 1
          -diff       [maximum weight]     // the maximum weight of edges in the tree, usually set as the same as m
          -N          [number of vectors]  // specify the number of vectors to use
          -query_size [SIZE]               // the number of queries to be performed
          -topk       [TOPK]               // topk
```
Example
```
./deltapq -dataset /data/local/pqdata/sift/ -ext fvecs -task query -m 8 -k 256 -h 1 -diff 8 -N 89656 -query_size 10 -topk 10 -debug
```
### Generate Groundtruth Files
Go to the dataset folder and 
```
mkdir groundtruth
```
Generate Groundtruth
```
./deltapq -dataset [path to the dataset folder] 
          -ext     [file type]          // specify the input file type, either fvecs or bvecs 
          -task groundtruth             // generate groundtruth
          -query_size [SIZE]            // the number of queries to be performed
          -topk       [TOPK]            // topk
          -N       [number of vectors]  // specify the number of vectors to use
```
Example
```
./pqtree -dataset /data/local/pqdata/sift/ -ext fvecs -task groundtruth -topk 10000 -query_size 1000 -N 1000000
```
