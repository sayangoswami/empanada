//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_BLOCKS_CUH
#define ASSEMBLY_V3_BLOCKS_CUH

#include "prelude.h"
#include "config.h"

struct TBlockList {
private:
    char* h_buf;
    const size_t BLKSZ;
    size_t curr_sz;
    std::deque<char*> datablocks;

public:
    TBlockList() : h_buf(nullptr), curr_sz(0), BLKSZ(Config::getInstance().base_blocksz) {
        CUDA_SAFE(cudaMallocHost((void**)&h_buf, BLKSZ));
    }
    ~TBlockList() {
        CUDA_SAFE(cudaFreeHost(h_buf));
    }
    /**
     * Bit-packs 4 bases into a byte
     * @param input[in] - input bases (must be a multiple of 4)
     * @param output[out] - packed output bases, 1/4th the size of the input vector
     */
    static void pack(const dvec<char>& input, dvec<char>& output) {
        if (input.size() % 4) LOG_ERR("Input array must be padded to a length which is a multiple of 4");

        auto _pack = [] __host__ __device__ (u4 i) {
            auto c = reinterpret_cast<char*>(&i);
            char b =(char)((c[0]>>1)&3)<<6 |
                    (char)((c[1]>>1)&3)<<4 |
                    (char)((c[2]>>1)&3)<<2 |
                    (char)((c[3]>>1)&3);
            return b;
        };

        auto in = reinterpret_cast<const u4*>(thrust::raw_pointer_cast(input.data()));
        const size_t n = input.size();
        output.resize(n/4);

        thrust::transform(thrust::device, in, in + n/4, output.begin(), _pack);
    }

    /**
     * Unpack the bits of the input array to bases in the output array
     * @param in[in] - the array containing the input bits
     * @param n[in] - the length (in bytes) of the input array
     * @param output[out] - the 4n-length vector used to store the decompressed characters
     */
    static void unpack(const dvec<char>& input, char *output) {
        expect(((uintptr_t)output) % 4 == 0);
        auto _unpack = [] __host__ __device__ (char b) {
            const char *S = "ACTG";
            u4 i = 0;
            auto c = reinterpret_cast<char*>(&i);
            c[0] = S[(char)((b&0xc0)>>6)];
            c[1] = S[(char)((b&0x30)>>4)];
            c[2] = S[(char)((b&0x0c)>>2)];
            c[3] = S[(char)((b&0x03)>>0)];
            return i;
        };

        auto out = reinterpret_cast<u4*>(output);
        thrust::transform(thrust::device, input.begin(), input.end(), out, _unpack);
    }
    void append(char* d_bases, size_t count) {
        while (count) {
            if (curr_sz == BLKSZ) {
                dvec<char> d_in(BLKSZ), d_out;
                CUDA_SAFE(cudaMemcpy(thrust::raw_pointer_cast(d_in.data()), h_buf, BLKSZ, cudaMemcpyHostToDevice));
                pack(d_in, d_out);
                expect(d_out.size() == BLKSZ/4);
                char* data = new char[BLKSZ/4];
                CUDA_SAFE(cudaMemcpy(data, thrust::raw_pointer_cast(d_out.data()), BLKSZ/4, cudaMemcpyDeviceToHost));
                datablocks.push_back(data);
                curr_sz = 0;
            }

            size_t n_copy = MIN(count, BLKSZ - curr_sz);
            CUDA_SAFE(cudaMemcpy(h_buf + curr_sz, d_bases, n_copy, cudaMemcpyDeviceToHost));
            curr_sz += n_copy;
            count -= n_copy;
            d_bases += n_copy;
        }
    }

    void copy_to_device(char* dst) {
        expect(((uintptr_t)dst) % 4 == 0);
        size_t off = 0;
        for (char* data : datablocks) {
            dvec<char> d_in(BLKSZ/4);
            CUDA_SAFE(cudaMemcpy(thrust::raw_pointer_cast(d_in.data()), data, BLKSZ/4, cudaMemcpyHostToDevice));
            unpack(d_in, dst + off);
            off += BLKSZ;
        }
        if (curr_sz) {
            CUDA_SAFE(cudaMemcpy(dst + off, h_buf, curr_sz, cudaMemcpyHostToDevice));
        }
    }

    const char& operator[](size_t i) const {
        if (datablocks.empty()) {
            if (i >= curr_sz)
                LOG_ERR("Out of bounds: i = %zd, curr_sz = %zd.", i, curr_sz);
            else return h_buf[i];
        }
        else {
            const size_t j = i / BLKSZ;
            if (j == datablocks.size()) { /// unpacked block
                if (i % BLKSZ >= curr_sz)
                    LOG_ERR("Out of bounds: i = %zd, BLKSZ = %zd, j = %zd, curr_sz = %zd.", i, BLKSZ, j, curr_sz);
                else return h_buf[i % BLKSZ];
            }
            else if (j < datablocks.size()) {        /// packed block
                const size_t k = (i/4) % (BLKSZ/4), shft = 6 - 2 * (i % 4);
                auto b = datablocks[j][k];
                const char *S = "ACTG";
                return S[(b >> shft) & 3];
            }
            else LOG_ERR("Out of bounds.");
        }
    }
};

struct TBases {

    unsigned n_reads, read_length;

    TBases() : n_reads(0), read_length(0) {}

    ~TBases() {
        for (auto b : blockLists) delete b;
    }

    void append(dvec<char>& d_bases, u4 nrows, u4 ncols) {
        TIMER_LOCAL_START;
        if (!n_reads) init(nrows);
        expect(nrows == read_length);
        expect(d_bases.size() == nrows * ncols);
        char* data = thrust::raw_pointer_cast(d_bases.data());

        for (int i = 0; i < nrows; ++i) {
            blockLists[i]->append(data, ncols);
            data += ncols;
        }
        n_reads += ncols;
        TIMER_LOCAL_STOP_W_LABEL("b_append");
    }

    /**
    * copy the i-th bases of all reads (in a 0-based indexing scheme)
    */
    void copy_to_device(char* dst, int i) {
        TIMER_LOCAL_START;
        expect(((uintptr_t)dst) % 4 == 0);
        blockLists[i]->copy_to_device(dst);
        TIMER_LOCAL_STOP_W_LABEL("b_copy2dev");
    }

    const TBlockList& operator[](size_t i) const {
        return *blockLists[i];
    }

private:

    std::vector<TBlockList*> blockLists;

    void init(u4 nrows) {
        read_length = nrows;
        expect(blockLists.empty());
        for (int i = 0; i < read_length; ++i) blockLists.push_back(new TBlockList());
    }
};

#endif //ASSEMBLY_V3_BLOCKS_CUH
