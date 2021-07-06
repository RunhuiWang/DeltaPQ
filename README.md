# DeltaPQ
DeltaPQ: Lossless Product Quantization Code Compression for High Dimensional Similarity Search
http://www.vldb.org/pvldb/vol13/p3603-wang.pdf

## How to compile

### Prerequisite

1. OpenCV

2. Boost (required components: timer chrono system program_options)

To compile
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
          -fvecs or -bvecs  // specify the input file type, either fvecs or bvecs
          -task learn       // learn codewords for PQ
          -m 8              // the number of sub-dimensions in PQ
          -k 256            // the number of centroids in each sub-dimension in PQ
          -train_size [training size] // the number of vectors used for learning codewords
```
Example
```
./pqtree -dataset /data/local/pqdata/sift/ -fvecs -task learn -m 8 -k 256 - train_size 10000
```
2. Encode
```
./pqtree  -dataset [path to the dataset folder] 
          -fvecs or -bvecs  // specify the input file type, either fvecs or bvecs
          -task encode      // encode PQ codes
          -m 8              // the number of sub-dimensions in PQ
          -k 256            // the number of centroids in each sub-dimension in PQ
```
Example
```
./pqtree -dataset /data/local/pqdata/sift/ -fvecs -task encode -m 8 -k 256
```
