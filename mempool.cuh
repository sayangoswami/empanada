//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_MEMPOOL_CUH
#define ASSEMBLY_V3_MEMPOOL_CUH

#include "prelude.h"
#include <map>
#include <queue>

class DevMempool {
    uintptr_t _buffer = 0;
    std::map<uintptr_t, uintptr_t> allocations;
public:
    const size_t ALLOCATION_SZ;
    explicit DevMempool(size_t size) : ALLOCATION_SZ(size) {
        expect(ALLOCATION_SZ > 0);
        LOG_DEBUG(LOW, "Creating DevMempool of size %zd.", ALLOCATION_SZ);
        void *b;
        CUDA_SAFE(cudaMalloc(&b, ALLOCATION_SZ));
        _buffer = (uintptr_t) b;
    }
    ~DevMempool() {
        LOG_DEBUG(LOW, "Destroying DevMempool of size %zd.", ALLOCATION_SZ);
        allocations.clear();
        CUDA_SAFE(cudaFree((void *)_buffer));
        _buffer = 0;
    }
    void * reserve(size_t n) {
        if (allocations.empty()) {
            allocations[_buffer] = _buffer + n;
            LOG_DEBUG(LOW, "Reserved %zd bytes at 0.", n);
            return (void *)_buffer;
        } else {
            auto it = allocations.rbegin();
            auto addr = (uintptr_t)alignup(it->second, 8);
            if (addr + n >= _buffer + ALLOCATION_SZ) {
                LOG_ERR("Out of memory. Have %zd bytes, requesting %zd bytes at %lu.",
                        ALLOCATION_SZ - addr, n, addr - _buffer);
            }
            else {
                allocations[addr] = addr + n;
                LOG_DEBUG(LOW, "Reserved %zd bytes at %lu.", n, addr - _buffer);
                return (void *)addr;
            }
        }
    }
    void release(void * addr) {
        LOG_DEBUG(LOW, "Released %zd bytes at %lu.",
                  allocations[(uintptr_t)addr] - (uintptr_t)addr, (uintptr_t)addr - _buffer);
        allocations.erase((uintptr_t)addr);
    }
};

template <typename T>
class HostBlockpool {
    std::queue<T*> queue;
public:
    const size_t ALLOCATION_CT;
    explicit HostBlockpool(size_t count): ALLOCATION_CT(count) {
        LOG_DEBUG(LOW, "Creating host block-pool");
        expect(ALLOCATION_CT > 0);
    }
    ~HostBlockpool() {
        while (!queue.empty()) {
            T *block = queue.front();
            queue.pop();
            delete[] block;
        }
        LOG_DEBUG(LOW, "Destroying host block-pool");
    }
    T* reserve() {
        T *block;
        if (queue.empty()) block = new T[ALLOCATION_CT];
        else {
            block = queue.front();
            queue.pop();
        }
        return block;
    }

    void release(T* block) {
        queue.push(block);
    }

};

typedef Singleton<DevMempool, size_t> Dmp;
typedef Singleton1<HostBlockpool, KeyT, size_t> Kmp;
typedef Singleton1<HostBlockpool, ValT, size_t> Vmp;


#endif //ASSEMBLY_V3_MEMPOOL_CUH
