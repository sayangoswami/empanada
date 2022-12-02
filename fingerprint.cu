//
// Created by sayan on 20-09-2022.
//

#include "assembly.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_output_iterator.h>

/// disable all prints (comment the following couple of lines to print)
#undef PRINT
#define PRINT(x) if(false)

unsigned h_modulos1[MAX_RD_LEN+1];
unsigned h_modulos2[MAX_RD_LEN+1];
__constant__ unsigned d_modulos1[MAX_RD_LEN+1];
__constant__ unsigned d_modulos2[MAX_RD_LEN+1];

/**
 * Generate a lookup table for sigma^i for i in [0, read-length]
 */
void precomputeModulos() {
    h_modulos1[0] = 1, h_modulos2[0] = 1;
    for (int i = 1; i <= MAX_RD_LEN; ++i) {
        h_modulos1[i] = (h_modulos1[i-1] * RADIX1) % PRIME1;
        h_modulos2[i] = (h_modulos2[i-1] * RADIX2) % PRIME2;
    }
    CUDA_SAFE(cudaMemcpyToSymbol(d_modulos1, h_modulos1, (MAX_RD_LEN + 1) * sizeof(unsigned)));
    CUDA_SAFE(cudaMemcpyToSymbol(d_modulos2, h_modulos2, (MAX_RD_LEN + 1) * sizeof(unsigned)));
}

struct FingerprintMul {
    __host__ __device__
    thrust::tuple<u1, u8> operator()(thrust::tuple<u1, u8> t0, thrust::tuple<u1, u8> t1) {
        u1 la = thrust::get<0>(t0), lb = thrust::get<0>(t1);
        uint64_t a = thrust::get<1>(t0), b = thrust::get<1>(t1);
        uint64_t
        fp1a = HIGH32(a),
        fp1b = HIGH32(b),
        fp2a = LOW32(a),
        fp2b = LOW32(b);
#ifdef __CUDA_ARCH__
        uint64_t newfp1 = (((fp1a * d_modulos1[lb]) % PRIME1) + fp1b) % PRIME1;
        uint64_t newfp2 = (((fp2a * d_modulos2[lb]) % PRIME2) + fp2b) % PRIME2;
#else
        uint64_t newfp1 = (((fp1a * h_modulos1[lb]) % PRIME1) + fp1b) % PRIME1;
        uint64_t newfp2 = (((fp2a * h_modulos2[lb]) % PRIME2) + fp2b) % PRIME2;
#endif
        u1 newlen = la + lb;
        u8 newfp = APPEND64(newfp1, newfp2);
        return thrust::make_tuple(newlen, newfp);
    }
};

struct EncodeBases : thrust::unary_function<char, thrust::tuple<u1, u8>> {
    __host__ __device__ __forceinline__
    thrust::tuple<u1, u8> operator()(char c) const {
        u8 e = ENCODING(c);
        return thrust::make_tuple(1, APPEND64(e, e));
    }
};

/**
 * Generates (fingerprint, read-id) tuples from reads
 * @param[in] start_id - The read-id offset (including complements) for this batch of reads
 * @param[in] reads - Input bases and their reverse complements
 * @param[in] stencil - Used to partition bases into reads
 * @param[out] fingerprints -
 * @param[out] read_ids -
 */
void generate_fingerprint_read_pairs(const u4 start_id, dvec<char>& reads, dvec<int>& stencil, dvec<u1>& fplens,
                                     dvec<uint64_t>& fingerprints, dvec<uint32_t>& read_ids) {
    TIMER_LOCAL_START;
    FingerprintMul fm;
    EncodeBases eb;

    size_t nbases = reads.size();
    expect(stencil.size() == nbases);

    fingerprints.resize(nbases);
    fplens.resize(nbases);

    auto tbegin = thrust::make_zip_iterator(thrust::make_tuple(fplens.begin(), fingerprints.begin()));

    /// generate fingerprints
    thrust::inclusive_scan_by_key(stencil.begin(), stencil.end(),
                                  thrust::make_transform_iterator(reads.begin(), eb),
                                  tbegin, thrust::equal_to<int>(), fm);

    /// in the stencil, mark partition-ends (last position in each read) with non-zero values
    read_ids.resize(nbases);
    thrust::copy(stencil.begin(), stencil.end(), read_ids.begin());
    thrust::adjacent_difference(stencil.rbegin(), stencil.rend(), stencil.rbegin());

    /// only keep fingerprints at the end of partitions
    auto isNotPartitionEnd = [] __host__ __device__ (const thrust::tuple<uint64_t, int, uint32_t>& t) {
        return (thrust::get<1>(t) == 0);
    };
    auto in_b = thrust::make_zip_iterator(thrust::make_tuple(fingerprints.begin(), stencil.begin(), read_ids.begin()));
    auto in_e = thrust::make_zip_iterator(thrust::make_tuple(fingerprints.end(), stencil.end(), read_ids.end()));
    auto out_e = thrust::remove_if(in_b, in_e, isNotPartitionEnd);
    size_t n = out_e - in_b;
    fingerprints.resize(n);
    stencil.resize(n);
    read_ids.resize(n);

    /// transform read-ids from line numbers to paired global read-ids
    thrust::sort_by_key(read_ids.begin(), read_ids.end(), fingerprints.begin());
    thrust::sequence(read_ids.begin(), read_ids.end(), start_id);
    TIMER_LOCAL_STOP;
}

/**
 * Transposes a batch of reads
 * @param[in] reads - d-vector of reads without newlines
 * @param[in] lengths  - d-vector of read lengths
 * @param[out] transposed - d-vector to store the transposed bases
 * @param[out] nrows - number of rows in the transposed matrix (read length)
 * @param[out] ncols - number of columns in the transposed matrix (the number of reads)
 */
void transpose_reads(dvec<char>& reads, dvec<int>& lengths, dvec<char>& transposed, int& nrows, int& ncols) {
    TIMER_LOCAL_START;
    const size_t n = reads.size();
    int nreads = lengths.size();
    expect(nreads > 0);
    int len = nrows = lengths[0];
    auto is_equal = [len] __host__ __device__ (const int i) {
        return i == len;
    };
    if (!thrust::all_of(lengths.begin(), lengths.end(), is_equal))
        LOG_ERR("All reads must be of equal length.");

    ncols = nreads;

    auto transpose_index = [nrows, ncols] __host__ __device__(const int i) {
        return i / nrows + (i % nrows) * ncols;
    };

    thrust::counting_iterator<size_t> indices(0);

    size_t n_out = nrows * ncols;
    transposed.resize(n_out);
    thrust::fill(transposed.begin(), transposed.end(), 0);

    thrust::scatter(reads.begin(), reads.end(),
                    thrust::make_transform_iterator(indices, transpose_index), transposed.begin());

    TIMER_LOCAL_STOP;
}