//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_CQUEUE_CUH
#define ASSEMBLY_V3_CQUEUE_CUH

#include "prelude.h"
#include "singleton.h"

template <typename T>
struct CQueue {
    struct Block {
        T *data;
        size_t offset, count;

        Block() : offset(0), count(0) {
            data = Singleton1<HostBlockpool, T, size_t>::getInstance()->reserve();
        }

        ~Block() {
            Singleton1<HostBlockpool, T, size_t>::getInstance()->release(data);
        }
    };

private:
    size_t count;
    std::deque<Block> blocks;

public:
    const size_t CQ_MAXCOUNT;
    CQueue() : count(0), CQ_MAXCOUNT(Config::getInstance().kv_chunksz) {
        expect(CQ_MAXCOUNT > 0);
    }

    ~CQueue() {
        expect(blocks.empty());
    }
    /**
     * @return the number of elements in the queue
     */
    size_t size() { return count; }

    void clear() {
        while (!blocks.empty()) {
            Block& block = blocks.front();
            block.count = 0;
            blocks.pop_front();
        }
        count = 0;
    }
    /**
     * Bulk-push elements into the queue. Allocates memory of the current chunk is full
     * @param src - source
     * @param n_elements - number of elements
     * @param kind -
     */
    void push_back(T* src, size_t n_elements, cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
        TIMER_LOCAL_START;
        if (blocks.empty()) blocks.emplace_back();
        while (n_elements) {
            if (blocks.back().count == CQ_MAXCOUNT) blocks.emplace_back();
            Block& block = blocks.back();
            size_t n = MIN(CQ_MAXCOUNT - block.count, n_elements);
            CUDA_SAFE(cudaMemcpy(block.data + block.count, src, n * sizeof(T), kind));
            src += n, n_elements -= n, block.count += n, count += n;
        }
        if (kind == cudaMemcpyDeviceToHost) TIMER_LOCAL_STOP_W_LABEL("memcpy_d2h");
        else if (kind == cudaMemcpyHostToHost) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
    }
    /**
     * Bulk-pop elements from the queue. Free a chunk if able.
     * @param dst
     * @param n_elements
     * @param kind
     * @return the number of items popped
     */
    size_t pop_front(T* dst, size_t n_elements, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
        TIMER_LOCAL_START;
        size_t remaining = n_elements;
        while (remaining) {
            if (blocks.empty()) break;
            Block& block = blocks.front();
            size_t n = MIN(block.count, remaining);
            CUDA_SAFE(cudaMemcpy(dst, block.data + block.offset, n * sizeof(T), kind));
            dst += n, remaining -= n, block.offset += n, block.count -= n, count -= n;
            if (block.count == 0) blocks.pop_front();
        }
        if (kind == cudaMemcpyHostToDevice) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2d");
        else if (kind == cudaMemcpyHostToHost) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
        return n_elements - remaining;
    }

    size_t pop_front(CQueue<T> *dst, size_t n_elements) {
        TIMER_LOCAL_START;
        size_t remaining = n_elements;
        while (remaining) {
            if (blocks.empty()) break;
            Block& block = blocks.front();
            size_t n = MIN(block.count, remaining);
            dst->push_back(block.data + block.offset, n, cudaMemcpyHostToHost);
            remaining -= n, block.offset += n, block.count -= n, count -= n;
            if (block.count == 0) blocks.pop_front();
        }
        TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
        return n_elements - remaining;
    }
    /**
     * Get the i-th element
     * @param i index
     * @return the element at index i
     */
    const T& operator[](size_t i) const {
        if (i < count) return blocks[i / CQ_MAXCOUNT].data[i % CQ_MAXCOUNT];
        else LOG_ERR("Array index out of bounds.");
    }

    T& operator[](size_t i) {
        if (i < count) return blocks[i / CQ_MAXCOUNT].data[i % CQ_MAXCOUNT];
        else LOG_ERR("Array index out of bounds.");
    }

    void copyTo(CQueue<T> *dst) {
        TIMER_LOCAL_START;
        for (Block& block : blocks) {
            dst->push_back(block.data, block.count, cudaMemcpyHostToHost);
        }
        TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
    }

    void copyTo(T *dst, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
        TIMER_LOCAL_START;
        size_t off = 0;
        for (Block& block : blocks) {
            CUDA_SAFE(cudaMemcpy(dst + off, block.data, block.count * sizeof(T), kind));
            off += block.v.size();
        }
        if (kind == cudaMemcpyHostToDevice) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2d");
        else if (kind == cudaMemcpyHostToHost) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
    }

    void copyTo(T *dst, size_t src_offset, size_t n_elements, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
        TIMER_LOCAL_START;
        size_t dst_offset = 0, blkid = src_offset / CQ_MAXCOUNT;
        src_offset = src_offset % CQ_MAXCOUNT;

        while (n_elements) {
            Block& block = blocks[blkid];
            T *src = block.data + src_offset;
            size_t n = MIN(n_elements, block.count - src_offset);
            CUDA_SAFE(cudaMemcpy(dst + dst_offset, src, n * sizeof(T), kind));
            dst_offset += n;
            n_elements -= n;
            src_offset = 0;
            blkid++;
            if (n_elements) expect(blkid < blocks.size());
        }
        if (kind == cudaMemcpyHostToDevice) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2d");
        else if (kind == cudaMemcpyHostToHost) TIMER_LOCAL_STOP_W_LABEL("memcpy_h2h");
    }

};

#endif //ASSEMBLY_V3_CQUEUE_CUH
