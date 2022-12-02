//
// Created by sayan on 20-09-2022.
//

#include "assembly.cuh"
#include "cqutils.cuh"
#include "blocks.cuh"
#include "timer.h"
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/transform_output_iterator.h>

#include "kthread.h"

typedef KeyT F;
typedef ValT R;

/**
 * Fast modular exponent
 * @param a
 * @param n
 * @param modulo
 * @return result of (a raised to the power n) modulo p
 */
static unsigned modulo_exp(u8 a, unsigned n, unsigned p) {
    u8 res = 1;
    while (n > 0) {
        if (n & 1)
            res = (res * a) % p;
        a = (a * a) % p;
        n >>= 1;
    }
    return res;
}

u4 RADIX1_INV, RADIX2_INV;
size_t M;
void *d_buf;

#ifdef TEST_NO_REMOVE_USED
void *h_edge_buf;
#endif

extern unsigned h_modulos1[MAX_RD_LEN+1];
extern unsigned h_modulos2[MAX_RD_LEN+1];
extern __constant__ unsigned d_modulos1[MAX_RD_LEN+1];
extern __constant__ unsigned d_modulos2[MAX_RD_LEN+1];

/**
 * functor to transform the fingerprint of an l length string to that of its (l-1) length suffix
 */
struct NextSfx {
    const int len;

    __host__ __device__
    explicit NextSfx(int len): len(len) {}

    __host__ __device__ __forceinline__
    uint64_t operator()(const KeyT& key, const char& base) const {
        uint64_t fp1 = HIGH32(key), fp2 = LOW32(key);
#ifdef __CUDA_ARCH__
        fp1 = (fp1 + PRIME1 - (ENCODING(base) * d_modulos1[len-1]) % PRIME1) % PRIME1;
        fp2 = (fp2 + PRIME2 - (ENCODING(base) * d_modulos2[len-1]) % PRIME2) % PRIME2;
#else
        fp1 = (fp1 + PRIME1 - (ENCODING(base) * h_modulos1[len-1]) % PRIME1) % PRIME1;
        fp2 = (fp2 + PRIME2 - (ENCODING(base) * h_modulos2[len-1]) % PRIME2) % PRIME2;
#endif
        return APPEND64(fp1, fp2);
    }
};

struct PrevPfx {
    const uint32_t sigma1_inv, sigma2_inv;

    __host__ __device__
    PrevPfx(uint32_t sigma1_inv, uint32_t sigma2_inv):
            sigma1_inv(sigma1_inv), sigma2_inv(sigma2_inv) {}

    __host__ __device__ __forceinline__
    uint64_t operator()(const KeyT& key, const char& base) const {
        uint64_t fp1 = HIGH32(key), fp2 = LOW32(key);
        fp1 = (sigma1_inv * (fp1 - ENCODING(base))) % PRIME1;
        fp2 = (sigma2_inv * (fp2 - ENCODING(base))) % PRIME2;
        return APPEND64(fp1, fp2);
    }
};

void emit_edges(CQueue<F> **pfx_fingerprints_p, CQueue<R> **pfx_read_ids_p,
                CQueue<F> **sfx_fingerprints_p, CQueue<R> **sfx_read_ids_p,
                dvec<size_t>& lb, dvec<size_t>& ub, dvec<R>& from, dvec<R>& to,
                const int ovlp_len, FILE *fp);
void get_edges(R *d_sfx_rids, size_t n, R *d_pfx_rids,
               const dvec<u4>& lower_bounds, const dvec<u4>& upper_bounds, dvec<u4>& from, dvec<u4>& to);
size_t remove_used(F *fingerprints, R *read_ids, size_t n, dvec<R>& used_read_ids);
template <class BinaryFunction>
void regenerate_fingerprints(CQueue<F> **fingerprints_p, CQueue<R> **read_ids_p,
                             char *fwd_bases, char *rev_bases, BinaryFunction func);
void filter_bases(R *read_ids, const size_t r, char *input_bases, char *output_bases, bool reverse = false);

void print_false_positive_rate(CQueue<R> *read_ids, CQueue<u4> *counts, CQueue<u4> *offsets,
                               TBases *basesX, int start, int len);

size_t find_and_dump_all_edges(size_t nA, size_t nB, dvec<size_t>& lb, dvec<size_t>& ub, FILE *fp);

size_t write_edges_adj_list(R *h_sfx_rids, size_t n, R *h_pfx_rids, u4* lower_bounds, u4* upper_bounds,
                          FILE *fp, int ovlp_len);

/**
 * Initialize buffers for reduce phase
 * @param m The number of key-value pairs that can be sorted on the device
 */
void init_match(const size_t m) {
    TIMER_LOCAL_START;
    M = m;
    CUDA_SAFE(cudaMalloc((void**)&d_buf, m * 13));
    RADIX1_INV = modulo_exp(RADIX1, PRIME1-2, PRIME1);
    RADIX2_INV = modulo_exp(RADIX2, PRIME2-2, PRIME2);
#ifdef TEST_NO_REMOVE_USED
    CUDA_SAFE(cudaMallocHost(&h_edge_buf, m * 12));
#endif
    TIMER_LOCAL_STOP;
}

void write_vertices(CQueue<R>& read_ids, int read_len, FILE *fp) {
    /**
     * Write vertex (short-read) info
       Column	Field	    Type	    Regexp	            Description
       ----------------------------------------------------------------------------------
            1	RecordType	Character	S	                Record type
            2	Name	    String	    [!-)+-<>-~][!-~]*	Segment name
            3	Sequence	String	    \*\|[A-Za-z=.]+	    Optional nucleotide sequence
       Optional fields:
       Tag	Type	Description
       -------------------------------
       LN	i	    Segment length
     */
    for (u4 i = 0; i < read_ids.size(); ++i) {
        u4 r = read_ids[i];
        if (r % 2 == 0) fprintf(fp, "S\t%u\t*\tLN:i:%d\n", r/2, read_len);
    }
}

size_t write_edges_binary(const dvec<R>& from, const dvec<R>& to, FILE *fp, int ovlp_len) {
    TIMER_LOCAL_START;
    u8 n = from.size();
    expect(to.size() == n);
    hvec<u4> h_from = from;
    hvec<u4> h_to = to;

    fwrite(&n, sizeof n, 1, fp); /// number of edges
    fwrite(&ovlp_len, sizeof ovlp_len, 1, fp); /// overlap length
    fwrite(h_from.data(), sizeof(R), n, fp);
    fwrite(h_to.data(), sizeof(R), n, fp);
    TIMER_LOCAL_STOP;
    return n;
}

size_t write_edges_gfa(const dvec<R>& from, const dvec<R>& to, FILE *fp, int ovlp_len) {
    size_t n = from.size();
    expect(to.size() == n);
    hvec<u4> h_from = from;
    hvec<u4> h_to = to;

    for (size_t i = 0; i < n; ++i) {
        /**
         * Column	Field	    Type	    Regexp	                    Description
         ----------------------------------------------------------------------------------------------------------
                1	RecordType	Character	L	                        Record type
                2	From	    String	    [!-)+-<>-~][!-~]*	        Name of segment
                3	FromOrient	String	    +\|-	                    Orientation of From segment
                4	To	        String	    [!-)+-<>-~][!-~]*	        Name of segment
                5	ToOrient	String	    +\|-	                    Orientation of To segment
                6	Overlap	    String	    \*\|([0-9]+[MIDNSHPX=])+	Optional CIGAR string describing overlap
        */
        u4 u = h_from[i]/4;
        char u_orient = (h_from[i] % 2 == 1) ? '+' : '-';

        u4 v = h_to[i]/4;
        char v_orient = (h_to[i] % 2 == 1) ? '+' : '-';

        fprintf(fp, "L\t%u\t%c\t%u\t%c\t%dM\n", u, u_orient, v, v_orient, ovlp_len);
    }

    return n;
}

size_t write_edges(const dvec<R>& from, const dvec<R>& to, FILE *fp, int ovlp_len) {
    TIMER_LOCAL_START;
    size_t n = from.size();
    expect(to.size() == n);
    hvec<u4> h_from = from;
    hvec<u4> h_to = to;

    for (size_t i = 0; i < n; ++i)
        fprintf(fp, "%u,%u,%d\n", h_from[i], h_to[i], ovlp_len);

    TIMER_LOCAL_STOP;
    return n;
}

void build_graph(const int readlen, CQueue<F> *fingerprints, CQueue<R> *read_ids, TBases *basesX,
                 const std::string& outfilename) {
    const size_t r = fingerprints->size();
    expect(r % 2 == 0);
    expect(read_ids->size() == r);

    cq_sort_by_key(&read_ids, &fingerprints, M, d_buf);
    cq_sort_by_key(&fingerprints, &read_ids, M, d_buf);

#ifdef FPCHECK
    auto counts = new CQueue<u4>(), offsets = new CQueue<u4>();
    LOG_DEBUG(LOW, "Counting unique keys..");
    cq_count_unique(fingerprints, M, d_buf, counts);
    print_false_positive_rate(read_ids, counts, offsets, basesX, 0, basesX->read_length);
#endif

    cq_unique_by_key(&fingerprints, &read_ids, M, d_buf);
    size_t r1 = fingerprints->size();
    expect(r1 == read_ids->size());
    LOG_INFO("%zd duplicates removed.", r - r1);

#ifdef FPCHECK
    expect(counts->size() == r1);
#endif

    FILE *fp = fopen(outfilename.c_str(), "w");
    if (!fp) LOG_ERR("Could not open %s because %s", outfilename.c_str(), strerror(errno));
//    write_vertices(read_ids, readlen, n_uniq_fp)

    auto sfx_fingerprints = fingerprints;
    auto sfx_read_ids = read_ids;

    auto pfx_fingerprints = new CQueue<F>();
    auto pfx_read_ids = new CQueue<R>();

    sfx_fingerprints->copyTo(pfx_fingerprints);
    sfx_read_ids->copyTo(pfx_read_ids);

    char *d_bases_fwd, *d_bases_bwd;
    CUDA_SAFE(cudaMalloc((void**)&d_bases_fwd, r/2));
    CUDA_SAFE(cudaMalloc((void**)&d_bases_bwd, r/2));

    PrevPfx prevPfx(RADIX1_INV, RADIX2_INV);

    dvec<size_t> lb, ub;
    dvec<R> from, to;

    expect(r == basesX->n_reads * 2);
    expect(readlen == basesX->read_length);

    const int MIN_OVLP_LEN = Config::getInstance().min_ovlp;

    for (int round = 0; (round < readlen - MIN_OVLP_LEN) && (sfx_fingerprints->size() > 0); ++round) {
        LOG_DEBUG(HIGH, "Round %d ..", round + 1);
        NextSfx nextSfx(readlen - round);

        basesX->copy_to_device(d_bases_fwd, round);
        basesX->copy_to_device(d_bases_bwd, readlen - 1 - round);

        LOG_DEBUG(MED, "Regenerating Fingerprints of %d-length prefixes and suffixes..", readlen - 1 - round);
        regenerate_fingerprints(&sfx_fingerprints, &sfx_read_ids, d_bases_fwd, d_bases_bwd, nextSfx);
        regenerate_fingerprints(&pfx_fingerprints, &pfx_read_ids, d_bases_bwd, d_bases_fwd, prevPfx);

        /// sort prefixes and suffixes
        LOG_DEBUG(MED, "Sorting prefixes and suffixes..");
        cq_sort_by_key(&sfx_fingerprints, &sfx_read_ids, M, d_buf);
        cq_sort_by_key(&pfx_fingerprints, &pfx_read_ids, M, d_buf);

#ifdef FPCHECK
        LOG_DEBUG(LOW, "Counting unique suffix fingerprints..");
        counts->clear(); offsets->clear();
        cq_count_unique(sfx_fingerprints, M, d_buf, counts);
        print_false_positive_rate(sfx_read_ids, counts, offsets, basesX, round + 1, basesX->read_length - round - 1);

        LOG_DEBUG(LOW, "Counting unique prefix fingerprints..");
        counts->clear(); offsets->clear();
        cq_count_unique(pfx_fingerprints, M, d_buf, counts);
        print_false_positive_rate(pfx_read_ids, counts, offsets, basesX, 0, basesX->read_length - round - 1);
#endif

        /// search for matches, write edges and remove used vertices
        LOG_DEBUG(MED, "Searching for matches..");
        emit_edges(&pfx_fingerprints, &pfx_read_ids, &sfx_fingerprints, &sfx_read_ids,
                   lb, ub, from, to, readlen - 1 - round, fp);
    }

    fclose(fp);
}

#ifdef FPCHECK
#include <set>
typedef struct {
    CQueue<R> *read_ids;
    CQueue<u4> *counts, *offsets;
    TBases *basesX;
    size_t *n_uniq_rd, *n_uniq_fp;
    std::set<std::string> *readset;
    int start, len;
} fpckeck_global_t;

const char revc[8] = {0,'T',0,'G','A',0,0,'C'};

static void get_read(R rid, TBases *basesX, char *out) {
    const int len = basesX->read_length;
    int r = rid/2;
     expect(r < basesX->n_reads);
    if (rid % 2 == 0) {         /// original
        for (int i = 0; i < len; ++i)
            out[i] = basesX->operator[](i)[r];
    } else {                    /// reverse complement
        for (int i = 0; i < len; ++i)
            out[len - 1 - i] = basesX->operator[](i)[r];
        for (int i = 0; i < len; ++i)
            out[i] = revc[(out[i]&7)];
    }
}

static void fpcheck(void *_g, long i, int tid) {
    auto g = (fpckeck_global_t*)_g;
    auto count = g->counts->operator[](i);
    auto off = g->offsets->operator[](i);
    const size_t len = g->basesX->read_length;
    char buf[len];
    if (count > 1) {
        std::set<std::string>& reads = g->readset[tid];
        for (int j = 0; j < count; ++j) {
            get_read(g->read_ids->operator[](off + j), g->basesX, buf);
            reads.insert(std::string(buf + g->start, g->len));
        }
        g->n_uniq_rd[tid] += reads.size();
        g->n_uniq_fp[tid]++;
        reads.clear();
    }
    else if (count == 1) g->n_uniq_rd[tid]++, g->n_uniq_fp[tid]++;
}

void print_false_positive_rate(CQueue<R> *read_ids, CQueue<u4> *counts, CQueue<u4> *offsets, TBases *basesX, int start, int len) {
    LOG_DEBUG(LOW, "Getting offsets from counts.");
    counts->copyTo(offsets);
    (*offsets)[0] = 0;
    for (size_t i = 1; i < offsets->size(); ++i) (*offsets)[i] = (*offsets)[i-1] + (*counts)[i-1];

    LOG_DEBUG(LOW, "Obtaining collisions..");
    const size_t nt = 10;
    fpckeck_global_t g = { read_ids, counts, offsets, basesX, new size_t[nt](), new size_t[nt](),
            new std::set<std::string>[nt](), start, len};
    kt_for(nt, fpcheck, &g, offsets->size());
    size_t n_uniq_rd = 0, n_uniq_fp = 0;
    for (int i = 0; i < nt; ++i) n_uniq_rd += g.n_uniq_rd[i], n_uniq_fp += g.n_uniq_fp[i];
    delete[] g.n_uniq_rd; delete[] g.n_uniq_fp; delete[] g.readset;
    LOG_DEBUG(MED, "Collision rate = %f%%", (1.0 - n_uniq_fp * 1.0 / n_uniq_rd) * 100.0);

}
#endif

void emit_edges(CQueue<F> **pfx_fingerprints_p, CQueue<R> **pfx_read_ids_p,
                CQueue<F> **sfx_fingerprints_p, CQueue<R> **sfx_read_ids_p,
                dvec<size_t>& lb, dvec<size_t>& ub, dvec<R>& from, dvec<R>& to,
                const int ovlp_len, FILE *fp) {

    auto pfx_fingerprints = *pfx_fingerprints_p;
    auto pfx_read_ids = *pfx_read_ids_p;
    auto sfx_fingerprints = *sfx_fingerprints_p;
    auto sfx_read_ids = *sfx_read_ids_p;

    const size_t N_P = pfx_fingerprints->size();
    expect(N_P == pfx_read_ids->size());
    const size_t N_S = sfx_fingerprints->size();
    expect(N_S == sfx_read_ids->size());

    size_t ne = 0;

    std::vector<std::pair<size_t, size_t>> partition_sizes;
    cq_get_search_partitions(*sfx_fingerprints, *pfx_fingerprints, M, partition_sizes);

    auto new_sfx_fingerprints = new CQueue<F>(), new_pfx_fingerprints = new CQueue<F>();
    auto new_sfx_read_ids = new CQueue<R>(), new_pfx_read_ids = new CQueue<R>();

    auto d_sfp = (F*)d_buf, d_pfp = d_sfp + M/2;

#ifdef TEST_NO_REMOVE_USED
    auto h_sfx_rid = (R*)h_edge_buf, h_pfx_rid = h_sfx_rid + M/2;
    auto h_lower_bound = (u4*)(h_pfx_rid + M/2), h_upper_bound = h_lower_bound + M/2;
#else
    auto d_srid = (R*)(d_pfp + M/2), d_prid = d_srid + M/2;
#endif

    for (auto p : partition_sizes) {
        size_t nA = p.first, nB = p.second, rc;
        if (nA && nB) {
            rc = sfx_fingerprints->pop_front(d_sfp, nA); expect(rc == nA);
            rc = pfx_fingerprints->pop_front(d_pfp, nB); expect(rc == nB);

#ifdef TEST_NO_REMOVE_USED
            rc = sfx_read_ids->pop_front(h_sfx_rid, nA, cudaMemcpyHostToHost); expect(rc == nA);
            rc = pfx_read_ids->pop_front(h_pfx_rid, nB, cudaMemcpyHostToHost); expect(rc == nB);

            new_sfx_fingerprints->push_back(d_sfp, nA);
            new_sfx_read_ids->push_back(h_sfx_rid, nA, cudaMemcpyHostToHost);
            new_pfx_fingerprints->push_back(d_pfp, nB);
            new_pfx_read_ids->push_back(h_pfx_rid, nB, cudaMemcpyHostToHost);

            ne += find_and_dump_all_edges(nA, nB, lb, ub, fp);
#else
            lb.resize(nA); ub.resize(nA);

            TIMER_GLOBAL_RESET;
            thrust::lower_bound(thrust::device, d_pfp, d_pfp + nB, d_sfp, d_sfp + nA, lb.begin());
            TIMER_GLOBAL_RECORD("Thrust lower bound");
            thrust::upper_bound(thrust::device, d_pfp, d_pfp + nB, d_sfp, d_sfp + nA, ub.begin());
            TIMER_GLOBAL_RECORD("Thrust upper bound");

            rc = sfx_read_ids->pop_front(d_srid, nA); expect(rc == nA);
            rc = pfx_read_ids->pop_front(d_prid, nB); expect(rc == nB);

            get_edges(d_srid, nA, d_prid, lb, ub, from, to);
            ne += write_edges_binary(from, to, fp, ovlp_len);

            rc = remove_used(d_sfp, d_srid, nA, from);
            new_sfx_fingerprints->push_back(d_sfp, rc);
            new_sfx_read_ids->push_back(d_srid, rc);
            rc = remove_used(d_pfp, d_prid, nB, to);
            new_pfx_fingerprints->push_back(d_pfp, rc);
            new_pfx_read_ids->push_back(d_prid, rc);

#endif
        }
        else if (nA) {

            rc = sfx_fingerprints->pop_front(new_sfx_fingerprints, nA); expect(rc == nA);
            rc = sfx_read_ids->pop_front(new_sfx_read_ids, nA); expect(rc == nA);
        }
        else {

            rc = pfx_fingerprints->pop_front(new_pfx_fingerprints, nB); expect(rc == nB);
            rc = pfx_read_ids->pop_front(new_pfx_read_ids, nB); expect(rc == nB);
        }
    }

    delete pfx_fingerprints; delete pfx_read_ids;
    delete sfx_fingerprints; delete sfx_read_ids;

    *pfx_fingerprints_p = new_pfx_fingerprints;
    *pfx_read_ids_p = new_pfx_read_ids;
    *sfx_fingerprints_p = new_sfx_fingerprints;
    *sfx_read_ids_p = new_sfx_read_ids;

    LOG_INFO("%zd edges created.", ne);
    LOG_INFO("%zd suffixes removed from next round.", N_S - (*sfx_fingerprints_p)->size());
    LOG_INFO("%zd prefixes removed from next round.", N_P - (*pfx_fingerprints_p)->size());
}

#ifdef TEST_NO_REMOVE_USED
size_t find_and_dump_all_edges(size_t nA, size_t nB, dvec<size_t>& lb, dvec<size_t>& ub, FILE *fp) {
    auto d_sfp = (F*)d_buf, d_pfp = d_sfp + M/2;
    auto d_srid = (R*)(d_pfp + M/2), d_prid = d_srid + M/2;

    auto h_sfx_rid = (R*)h_edge_buf, h_pfx_rid = h_sfx_rid + M/2;
    auto h_lower_bound = (size_t *)(h_pfx_rid + M/2), h_count = h_lower_bound + M/2;

    /// find occurrences of prefixes in list of suffixes
    lb.resize(nB); ub.resize(nB);
    thrust::lower_bound(thrust::device, d_sfp, d_sfp + nA, d_pfp, d_pfp + nB, lb.begin());
    thrust::upper_bound(thrust::device, d_sfp, d_sfp + nA, d_pfp, d_pfp + nB, ub.begin());

    /// filter prefixes which occur in the list of suffixes
    auto it_input = thrust::make_zip_iterator(thrust::make_tuple(d_pfp, d_prid));
    auto it_stencil = thrust::make_zip_iterator(thrust::make_tuple(ub.begin(), lb.begin()));
    auto is_not_there = [] __host__ __device__ (const thrust::tuple<size_t, size_t>& t) {
        return (thrust::get<0>(t) == thrust::get<1>(t));
    };
    auto new_end = thrust::remove_if(thrust::device, it_input, it_input + nB, it_stencil, is_not_there);
    nB = new_end - it_input;

    /// find occurrences of suffixes in list of suffixes
    lb.resize(nA); ub.resize(nA);
    thrust::lower_bound(thrust::device, d_pfp, d_pfp + nB, d_sfp, d_sfp + nA, lb.begin());
    thrust::upper_bound(thrust::device, d_pfp, d_pfp + nB, d_sfp, d_sfp + nA, ub.begin());

    /// filter suffixes which occur in the list of prefixes
    it_input = thrust::make_zip_iterator(thrust::make_tuple(d_sfp, d_srid));
    it_stencil = thrust::make_zip_iterator(thrust::make_tuple(ub.begin(), lb.begin()));
    new_end = thrust::remove_if(thrust::device, it_input, it_input + nA, it_stencil, is_not_there);
    nA = new_end - it_input;

    /// find edge counts and stats
    auto& counts = ub;
    thrust::minus<size_t> op;
    thrust::transform(thrust::device, ub.begin(), ub.end(), lb.begin(), counts.begin(), op);
    size_t ne = thrust::reduce(counts.begin(), counts.end());
    size_t mi = thrust::reduce(counts.begin(), counts.end(), 0, thrust::maximum<u4>());

    /// copy to host and write edges
    LOG_INFO("Edge stats: Sources = %zd, Sinks = %zd, Edges = %zd, Max-outdegree = %zd", nA, nB, ne, mi);
    CUDA_SAFE(cudaMemcpy(h_sfx_rid, d_srid, nA * sizeof(u4), cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(h_pfx_rid, d_prid, nB * sizeof(u4), cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(h_lower_bound, thrust::raw_pointer_cast(lb.data()), nA * sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaMemcpy(h_count, thrust::raw_pointer_cast(counts.data()), nA * sizeof(size_t), cudaMemcpyDeviceToHost));
    fwrite(&nA, sizeof(nA), 1, fp); /// number of vertices with outgoing edges
    fwrite(&nB, sizeof(nB), 1, fp); /// number of vertices with incoming edges
    fwrite(&ne, sizeof(ne), 1, fp); /// number of edges
    fwrite(&mi, sizeof(mi), 1, fp); /// maximum in-degree
    fwrite(h_sfx_rid, sizeof(u4), nA, fp);
    fwrite(h_pfx_rid, sizeof(u4), nB, fp);
    fwrite(h_lower_bound, sizeof(size_t), nA, fp);
    fwrite(h_count, sizeof(size_t), nA, fp);
    LOG_INFO("Finished writing edges to file.");

    return ne;
}

void dump_reads_with_same_fingerprints(size_t nA, size_t nB, FILE *fp) {
    auto d_sfp = (F*)d_buf, d_pfp = d_sfp + M/2;
    auto d_srid = (R*)(d_pfp + M/2), d_prid = d_srid + M/2;

    /// count by keys - suffixes
    dvec<F> keys(nA); dvec<u4> counts(nA), offsets(nA);
    auto new_end = thrust::reduce_by_key(thrust::device, d_sfp, d_sfp + nA,
                                         thrust::constant_iterator<size_t>(1), keys.begin(), counts.begin());
    size_t n_uniq = new_end.first - keys.begin();
    thrust::exclusive_scan(counts.begin(), counts.begin() + nuniq, offsets.begin());

    /// only keep keys with counts > 1
    auto it1_begin = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), counts.begin(), offsets.begin()));
    auto is_less_than_2 = [] __host__ __device__ (const thrust::tuple<F, u4, u4>& t) {
        return (thrust::get<1>(t) < 2);
    };
    auto it1_end = thrust::remove_if(thrust::device, it1_begin, it1_begin + nA, is_less_than_2);
    size_t n_dup = it1_end - it1_begin;


    auto h_sfx_rid = (R*)h_edge_buf, h_pfx_rid = h_sfx_rid + M/2;
}

size_t write_edges_adj_list(R *h_sfx_rids, size_t n, R *h_pfx_rids, u4* lower_bounds, u4* upper_bounds,
                          FILE *fp, int ovlp_len) {
    size_t ne = 0;
    for (size_t i = 0; i < n; ++i) {
        if (upper_bounds[i] - lower_bounds[i] > 0) {
            fprintf(fp, "%u", h_sfx_rids[i]);
            for (u4 j = lower_bounds[i]; j < upper_bounds[i]; ++j)
                fprintf(fp, ",%u", h_pfx_rids[j]);
            fprintf(fp, "\n");
            ne += upper_bounds[i] - lower_bounds[i];
        }
    }
    return ne;
}

#endif

void get_edges(R *d_sfx_rids, size_t n, R *d_pfx_rids,
               const dvec<u4>& lower_bounds, const dvec<u4>& upper_bounds, dvec<u4>& from, dvec<u4>& to) {
    TIMER_LOCAL_START;
    /// for each source vertex (suffix), find number of edges and their offsets
    dvec<int> counts(n, 0), offsets(n, 0);
    thrust::minus<u4> op;
    thrust::transform(upper_bounds.begin(), upper_bounds.end(), lower_bounds.begin(), counts.begin(), op);
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets.begin());

    /// find the total number of edges and allocate space for output
    u4 ne = offsets.back() + counts.back();
    from.resize(ne);
    thrust::fill(from.begin(), from.end(), 0);
    to.resize(ne);

    /// scatter source vertex indexes according to their offsets
    dvec<u4> from_indices(ne);
    auto it = thrust::make_counting_iterator(0);
    thrust::scatter_if(it, it + n, offsets.begin(), counts.begin(), from_indices.begin());

    /// copy source vertex indices for when a vertex has multiple outgoing edges
    thrust::inclusive_scan(from_indices.begin(), from_indices.end(), from_indices.begin(), thrust::maximum<int>());

    /// gather actual src read ids based on their indices
    thrust::gather(thrust::device, from_indices.begin(), from_indices.end(), d_sfx_rids, from.begin());

    /// scatter destination vertex indexes according to their offsets
    dvec<u4> to_indices(ne, 1);
    thrust::scatter_if(lower_bounds.begin(), lower_bounds.end(), offsets.begin(), counts.begin(), to_indices.begin());

    /// increment and copy dst index for when they have incoming edges from the same source
    thrust::inclusive_scan_by_key(from_indices.begin(), from_indices.end(), to_indices.begin(), to_indices.begin());

    /// gather actual dst read ids based on their indices
    thrust::gather(thrust::device, to_indices.begin(), to_indices.end(), d_pfx_rids, to.begin());

    /// remove self edges
    auto is_same_read = [] __host__ __device__ (const thrust::tuple<u4, u4>& t) {
        return (bool)(thrust::get<0>(t)/2 == thrust::get<1>(t)/2);
    };
    auto first = thrust::make_zip_iterator(thrust::make_tuple(from.begin(), to.begin()));
    auto new_end = thrust::remove_if(first, first + from.size(), is_same_read);
    size_t n1 = new_end - first;
    from.resize(n1);
    to.resize(n1);
    TIMER_LOCAL_STOP;
}

template <class BinaryFunction>
void regenerate_fingerprints(CQueue<F> **fingerprints_p, CQueue<R> **read_ids_p,
                             char *fwd_bases, char *rev_bases, BinaryFunction func) {
    TIMER_LOCAL_START;
    auto fingerprints = *fingerprints_p;
    auto read_ids = *read_ids_p;
    const size_t count = fingerprints->size();
    expect(count == read_ids->size());
    cq_sort_by_key(&read_ids, &fingerprints, M, d_buf);
    size_t src_offset = 0;

    auto tmp_fingerprints = new CQueue<F>();
    auto tmp_read_ids = new CQueue<R>();

    auto d_fp = (F*)d_buf;
    auto d_rid = (R*)(d_fp + M);
    auto d_base_buf = (char*)(d_rid + M);

    while (src_offset < count) {
        size_t np = MIN(count - src_offset, M);

        fingerprints->pop_front(d_fp, np);
        read_ids->pop_front(d_rid, np);

        /// rearrange tuples such that it consists of original read-ids followed by reverse complement ids
        TIMER_GLOBAL_RESET;
        auto isEven = [] __host__ __device__ (const thrust::tuple<R, F>& t) {
            R i = thrust::get<0>(t);
            return (bool)(!(i & 1));
        };
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_rid, d_fp));
        auto it_p = thrust::partition(thrust::device, begin, begin + np, isEven);
        size_t np1 = it_p - begin, np2 = np - np1;
        TIMER_GLOBAL_RECORD("Thrust partition in regenerate");

        /// sort both partitions by read ids
        thrust::sort_by_key(thrust::device, d_rid, d_rid + np1, d_fp);
        thrust::sort_by_key(thrust::device, d_rid + np1, d_rid + np, d_fp + np1);\
        TIMER_GLOBAL_RECORD("Thrust sort in regenerate");

        /// for each partition, filter bases and get next fingerprints
        filter_bases(d_rid, np1, fwd_bases, d_base_buf);
        thrust::transform(thrust::device, d_fp, d_fp + np1, d_base_buf, d_fp, func);
        TIMER_GLOBAL_RECORD("Thrust transform in regenerate");

        filter_bases(d_rid + np1, np2, rev_bases, d_base_buf, true);
        thrust::transform(thrust::device, d_fp + np1, d_fp + np, d_base_buf, d_fp + np1, func);
        TIMER_GLOBAL_RECORD("Thrust transform in regenerate");

        /// copy newly created fingerprints and associated read ids to host memory
        tmp_fingerprints->push_back(d_fp, np);
        tmp_read_ids->push_back(d_rid, np);

        src_offset += np;
    }

    delete fingerprints;
    delete read_ids;

    *fingerprints_p = tmp_fingerprints;
    *read_ids_p = tmp_read_ids;
    TIMER_LOCAL_STOP;
}

/** fixme - instead of gathering from all reads, only gather from a subset which corresponds to the range currently in device memory */
void filter_bases(R *read_ids, const size_t r, char *input_bases, char *output_bases, bool reverse) {
    TIMER_LOCAL_START;
    /// lambda to get read number from read id
    auto get_rid = [] __host__ __device__ (const u4& x) { return x/2;};

    /// lambda to get WC-complement base from original base
    auto revc = [] __host__ __device__ (const char c) {
#ifdef __CUDA_ARCH__
        return d_revc[(c&7)];
#else
        return h_revc[(c&7)];
#endif
    };

    auto it = thrust::make_transform_iterator(read_ids, get_rid);

    if (reverse) {
        thrust::gather(thrust::device, it, it + r,
                       thrust::make_transform_iterator(input_bases, revc), output_bases);
    }
    else thrust::gather(thrust::device, it, it + r, input_bases, output_bases);
    TIMER_LOCAL_STOP;
}

size_t remove_used(F *fingerprints, R *read_ids, size_t n, dvec<R>& used_read_ids) {
#ifdef TEST_NO_REMOVE_USED
    return n;
#else
    TIMER_LOCAL_START;
    dvec<bool> is_there(n, false);
    thrust::sort(used_read_ids.begin(), used_read_ids.end());
    thrust::binary_search(thrust::device, used_read_ids.begin(), used_read_ids.end(),
                          read_ids, read_ids + n, is_there.begin());

    auto first = thrust::make_zip_iterator(thrust::make_tuple(read_ids, fingerprints));
    auto new_end = thrust::remove_if(thrust::device, first, first + n, is_there.begin(), thrust::identity<bool>());
    size_t n1 = new_end - first;
    TIMER_LOCAL_STOP;
    return n1;
#endif
}

