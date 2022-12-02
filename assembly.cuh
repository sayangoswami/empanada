//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_ASSEMBLY_CUH
#define ASSEMBLY_V3_ASSEMBLY_CUH

#include "singleton.h"
#include "timer.h"
#include "config.h"
#include "mempool.cuh"
#include "cqueue.cuh"
#include "blocks.cuh"

/// switches and configurations
#define VERBOSE false
#define MAX_RD_LEN 255

///Pseudo Mersenne Prime numbers (closest to 2^k)
#define Q_24 16777213
#define Q_25 33554393
#define Q_26 67108859
#define Q_27 134217689
#define Q_28 268435399
#define Q_29 536870909
#define Q_30 1073741789
#define Q_31 2147483647

#define PRIME1 536870909
#define PRIME2 536870879
#define RADIX1 5
#define RADIX2 7

#define ENCODING(c) ((((c) >> 1) & 3) + 1)

///**
//* A key is a 64-bit unsigned integer
//* composed of an 8-byte length, a 27 byte fingerprint, and a 29 byte fingerprint
//*/
//#define KEY(length, fp_high, fp_low) (                              \
//                ((uint64_t)(length) << 56u) |                       \
//                ((((uint64_t)(fp_high)) & ((1<<27)-1)) << 29u) |    \
//                (((uint64_t)(fp_low)) & ((1<<29)-1))                \
//)
//
//#define LEN(key)    ((key) >> 56u)
//#define FP1(key)    (((key) >> 29u) & ((1<<27)-1))
//#define FP2(key)    ((key) & ((1<<29)-1))

#define HIGH32(x)   (((x) >> 32u) & 0xffffffffu)
#define LOW32(x)    ((x) & 0xffffffffu)
#define APPEND64(high32, low32) ((((high32) & 0xffffffffu) << 32u) | ((low32) & 0xffffffffu))

/// macros
#define PRINT(x) if(VERBOSE && (x))

#define PRINT_VECTOR(vector, format) do {    \
    LOG_INFO("%s:", #vector);                \
    for (auto i : (vector))                  \
        printf(format, i);                   \
    printf("\n");                            \
} while(0)

extern __constant__ char d_revc[8];
extern const char h_revc[8];

void precomputeModulos();

bool read_block(std::ifstream& file, size_t BLOCKSIZE,
                dvec<char>& input, dvec<int>& stencil, dvec<int>& lengths, dvec<int>& offsets);

void get_reverse_complements(dvec<char>& reads, dvec<int>& stencil, dvec<int>& lengths, dvec<int>& offsets);

void generate_fingerprint_read_pairs(u4 start_id, dvec<char>& reads, dvec<int>& stencil, dvec<u1>& fplens,
                                     dvec<KeyT>& fingerprints, dvec<ValT>& read_ids);

void transpose_reads(dvec<char>& reads, dvec<int>& lengths, dvec<char>& transposed, int& nrows, int& ncols);

void init_match(size_t m);

void build_graph(int readlen, CQueue<KeyT> *fingerprints, CQueue<ValT> *read_ids, TBases *basesX,
                 const std::string& outfilename);


#endif //ASSEMBLY_V3_ASSEMBLY_CUH
