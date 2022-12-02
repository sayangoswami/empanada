//
// Created by sayan on 20-09-2022.
//

#include "assembly.cuh"
#include <sys/stat.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

/// disable all prints (comment the following couple of lines to print)
#undef PRINT
#define PRINT(x) if(false)

__constant__ char d_revc[8];
const char h_revc[8] = {0,'T',0,'G','A',0,0,'C'};

/**
 * Parse lines (including newlines) and count them
 * @param input[in] - input array
 * @param stencil[out] - stencil to partition input into reads
 * @return - the number of lines parsed
 */
int parseLines(dvec<char>& input, dvec<int>& stencil) {
    auto isNewLine = [] __host__ __device__ (const char c) { return (int)(c == '\n'); };

    /// create a stencil for partitioning lines. Bases in the same lines have same IDs
    stencil.resize(input.size());
    thrust::transform_exclusive_scan(input.begin(), input.end(),
                                     stencil.begin(), isNewLine, 0, thrust::plus<int>());

    /// get the number of lines read
    int nlines = stencil.back();
    return nlines+1;
}

/**
 * Resize the input buffer so that it only consumes complete read blocks (including newlines)
 * @param input[in] - the input array on device
 * @param n_lines[in] - the number of lines in the input
 * @param stencil[in, out] - stencil to partition input into reads
 * @return the number of bytes discarded from the end of the buffer
 */
int resizeBuffer(dvec<char>& input, int& n_lines, dvec<int>& stencil) {
    /// calculate the number of bytes to seek back
    int nreads = (n_lines-1) / 4;
    n_lines = nreads * 4;

    auto isExtra = [n_lines] __host__ __device__ (const int x) {
        return x >= n_lines;
    };
    int seekback_count = thrust::count_if(stencil.begin(), stencil.end(), isExtra);

    /// resize buffers
    size_t n = input.size() - seekback_count;
    input.resize(n);
    stencil.resize(n);

    return seekback_count;
}

/**
 * Extract reads (including newlines) from fastq read blocks
 * @param input[in, out] - input array (the output is stored here)
 * @param stencil[in,out] - stencil used to partition lines/reads
 * @return number of reads extracted
 */
int extractReads(dvec<char>& input, dvec<int>& stencil) {
    /// only keep every 2nd line in a 4 line block
    auto isNotFastqRead = [] __host__ __device__ (const thrust::tuple<char, int>& t) {
        return thrust::get<1>(t) % 4 != 1;
    };
    auto in_b = thrust::make_zip_iterator(thrust::make_tuple(input.begin(), stencil.begin()));
    auto in_e = thrust::make_zip_iterator(thrust::make_tuple(input.end(), stencil.end()));
    auto out_e = thrust::remove_if(in_b, in_e, isNotFastqRead);
    size_t n = out_e - in_b;

    /// resize buffers
    input.resize(n); input.shrink_to_fit();
    stencil.resize(n); stencil.shrink_to_fit();

    /// get the number of lines read
    int n_lines = stencil.back() / 4;
    return n_lines + 1;
}

/**
 * Remove low-quality reads and new-lines.
 * This is an in-place operation
 * @param reads[in, out] - the vector of reads including newlines
 * @param stencil[in, out] - the vector of ids used to partition reads
 * @param lengths[out] - vector of lengths of the high-quality reads
 * @param offsets[out] - vector of offsets of the high-quality reads
 * @return the number of high-quality reads extracted
 */
int removeLQReads(dvec<char>& reads, dvec<int>& stencil, dvec<int>& lengths, dvec<int>& offsets) {
    int n_reads = lengths.size();
    int n_bases = reads.size();

    /// transform bases to ints, change N's to 9's and then check presence of 9s
    dvec<int> read_ids(n_reads);
    dvec<int> max_chars(n_reads);

    /**
     * lambda to mark N's in reads
     * Dec	Hex	Binary		Char
        ---------------------------------------------
        10	0A	00001010	Newline
        65	41	01000001	A
        67	43	01000011	C
        71	47	01000111	G
        84	54	01010100	T
        78	4E	01001110	N
     * Each char is encoded based on their b_7 and b_4 values, where b_i is the i-th bit from the right
     * This encodes newlines to 0s, valid bases to 1s, and Ns to 9s.
     */
    auto markNs = [] __host__ __device__ (const char c) {
        return (c >> 6) * (1 + (c & 8));
    };

    expect(stencil.size() == n_bases);
    /**
     * Find out which reads have Ns. Results in key-value pairs where key is read-id and value is either 9 or 1
     * depending on whether the read has N or not.
     */
    thrust::reduce_by_key(stencil.begin(), stencil.end(),
                          thrust::make_transform_iterator(reads.begin(), markNs),
                          read_ids.begin(), max_chars.begin(),
                          thrust::equal_to<char>(), thrust::maximum<char>());

    /// calculate read lengths
    thrust::reduce_by_key(stencil.begin(), stencil.end(),
                          thrust::make_constant_iterator(1),
                          read_ids.begin(), lengths.begin());

    /// calculate read offsets
    offsets.resize(n_reads);
    thrust::exclusive_scan(lengths.begin(), lengths.end(), offsets.begin());

    /// create another stencil where all bases belonging to an erroneous read are marked 9s
    dvec<int> stencil_new(n_bases);
    thrust::fill(stencil_new.begin(), stencil_new.end(), 0);
    thrust::scatter(max_chars.begin(), max_chars.end(), offsets.begin(), stencil_new.begin());
    thrust::inclusive_scan_by_key(stencil.begin(), stencil.end(),
                                  stencil_new.begin(), stencil_new.begin());

    /// only keep bases that are 1s
    auto isNotHighQualBase = [] __host__ __device__ (const thrust::tuple<char, int, int>& t) {
        return (thrust::get<2>(t) == 9) || (thrust::get<0>(t) == '\n');
    };
    auto in_b = thrust::make_zip_iterator(thrust::make_tuple(reads.begin(), stencil.begin(), stencil_new.begin()));
    auto in_e = thrust::make_zip_iterator(thrust::make_tuple(reads.end(), stencil.end(), stencil_new.end()));
    auto out_e = thrust::remove_if(in_b, in_e, isNotHighQualBase);
    size_t n = out_e - in_b;

    reads.resize(n);
    stencil.resize(n);
    stencil_new.resize(n);

    if (n) {
        /// recalculate read lengths
        auto new_end = thrust::reduce_by_key(stencil.begin(), stencil.end(),
                                             thrust::make_constant_iterator(1),
                                             read_ids.begin(), lengths.begin());
        n_reads = new_end.first - read_ids.begin();

        lengths.resize(n_reads);
        offsets.resize(n_reads);

        /// recalculate offsets
        thrust::exclusive_scan(lengths.begin(), lengths.end(), offsets.begin());

        return n_reads;
    }
    else {
        lengths.clear();
        offsets.clear();
        return 0;
    }
}

/**
 * Reads a block of input from the current file position.
 * It expects that the file pointer is at the start of some read.
 * @param[in] file - input file, must be opened in binary mode
 * @param[in] BLOCKSIZE - the max size of input data that can be processed on the device
 * @param[out] input - device vector for bases
 * @param[out] stencil - device vector for the stencil that partitions bases into reads
 * @param[out] lengths - device vector of read lengths
 * @param[out] offsets - device vector of read offsets
 * @return true of this was the last block, false otherwise
 */
bool read_block(std::ifstream& file, const size_t BLOCKSIZE,
                dvec<char>& input, dvec<int>& stencil, dvec<int>& lengths, dvec<int>& offsets) {
    TIMER_GLOBAL_RESET;
    LOG_INFO("Reading block..");
    bool done = false;
    hvec<char> h_input(BLOCKSIZE);
    file.read(h_input.data(), BLOCKSIZE);
    if (file.gcount() != BLOCKSIZE) done = true;
    if (file.gcount() == 0) {
        lengths.clear();
        return true;
    }
    expect(h_input.front() == '@');
    h_input.resize(file.gcount());
    input.resize(file.gcount());

    LOG_INFO("Parsing lines..");
    thrust::copy(h_input.begin(), h_input.end(), input.begin());
    TIMER_GLOBAL_RECORD("read_input");

    TIMER_GLOBAL_RESET;
    int n_lines = parseLines(input, stencil);
    LOG_DEBUG(LOW, "%d lines parsed from %zd characters.", n_lines, input.size());

    if (!done && (n_lines % 4 != 0 || h_input.back() != '\n')) {
        long nseek = resizeBuffer(input, n_lines, stencil);
        LOG_DEBUG(LOW, "Seeking back %d chars.", nseek);
        file.seekg(-nseek, std::ios::cur);
        expect(file.peek() == '@');
    }

    LOG_INFO("Extracting reads..");
    int n_reads = extractReads(input, stencil);
    expect(n_reads == n_lines/4);
    lengths.resize(n_reads);
    offsets.resize(n_reads);

    LOG_INFO("Removing Low-quality reads..");
    removeLQReads(input, stencil, lengths, offsets);

    TIMER_GLOBAL_RECORD("preprocess_input");
    return done;
}

/**
 * Get the reverse complements from a bunch of reads and push_back them to the input
 * @param reads[in,out] - Input bases. The reverse complement bases are appended to this vector
 * @param stencil[in,out] - Stencil used to partition bases into reads. The stencil of RC-reads are appended to this.
 * @param lengths[in,out] - Read lengths. The lengths of RC-reads are appended here.
 * @param offsets[in,out] - Read offsets. The offsets of RC-reads are appended here.
 */
void get_reverse_complements(dvec<char>& reads, dvec<int>& stencil, dvec<int>& lengths, dvec<int>& offsets) {
    TIMER_LOCAL_START;
    LOG_INFO("Calculating reverse complements..");
    const size_t n = reads.size();

    reads.resize(2*n);
    stencil.resize(2*n);

    auto reverse = [] __host__ __device__ (const thrust::tuple<char, int>& t) {
        char b = thrust::get<0>(t);
        int s = thrust::get<1>(t);

#ifdef __CUDA_ARCH__
        b = d_revc[(b&7)];
#else
        b = h_revc[(b&7)];
#endif
        s = s+1;
        return thrust::make_tuple(b, s);
    };

    auto rb = reads.begin();
    auto sb = stencil.begin();
    auto in_b = thrust::make_zip_iterator(thrust::make_tuple(rb, sb));

    auto out_e = thrust::make_zip_iterator(thrust::make_tuple(reads.end(), stencil.end()));
    auto r_out = thrust::make_reverse_iterator(out_e);

    auto in_r = thrust::make_transform_iterator(in_b, reverse);
    thrust::copy(in_r, in_r + n, r_out);

    const size_t r = lengths.size();
    lengths.resize(2 * r);
    offsets.resize(2 * r);
    thrust::copy(lengths.begin(), lengths.begin() + r, thrust::make_reverse_iterator(lengths.end()));
    thrust::exclusive_scan(lengths.begin(), lengths.end(), offsets.begin());

    TIMER_LOCAL_STOP;
}