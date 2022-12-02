//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_CQUTILS_H
#define ASSEMBLY_V3_CQUTILS_H

#include "assembly.cuh"
#include <queue>
#include <thrust/sort.h>

template<typename T>
static size_t cq_upper_bound(CQueue<T>& cQueue, size_t start, size_t end, const T& key) {
    expect(cQueue.size() >= end);
    size_t i;
    while (start < end) {
        i = (start + end) / 2;
        if (key < cQueue[i]) end = i;
        else start = i + 1;
    }
    return start;
}

template<typename T>
static size_t cq_lower_bound(CQueue<T>& cQueue, size_t start, size_t end, const T& key) {
    expect(cQueue.size() >= end);
    size_t i;
    while (start < end) {
        i = (start + end) / 2;
        if (key > cQueue[i]) start = i + 1;
        else end = i;
    }
    return start;
}

template <typename Key>
static void cq_get_merge_partitions(CQueue<Key>& keys_A, CQueue<Key>& keys_B, const size_t M,
                             std::vector<std::pair<size_t, size_t>>& partition_sizes) {
    TIMER_LOCAL_START;
    const size_t nA = keys_A.size(), nB = keys_B.size();
    size_t offA = 0, offB = 0;

    while (offA < nA && offB < nB) {
        size_t nPa = MIN(nA - offA, M / 2), nPb = MIN(nB - offB, M / 2);

        if (nPa < M / 2 && nPb < M / 2) {
            partition_sizes.emplace_back(nPa, nPb);
            offA += nPa, offB += nPb;
            break;
        }

        /// Case 1: all of A is smaller than B
        if (keys_A[offA + nPa - 1] <= keys_B[offB]) {
            partition_sizes.emplace_back(nPa, 0);
            offA += nPa;
        }
            /// Case 2: all of B is smaller than A
        else if (keys_B[offB + nPb - 1] <= keys_A[offA]) {
            partition_sizes.emplace_back(0, nPb);
            offB += nPb;
        }
            /// Case 3: No clear winner, need to merge
        else {
            /// Case a: A[na] < B[nb]
            if (keys_A[offA + nPa - 1] < keys_B[offB + nPb - 1]) {
                nPb = cq_upper_bound(keys_B, offB, offB + nPb, keys_A[offA + nPa - 1]) - offB;
            }

                /// Case a: B[nb] < A[na]
            else if (keys_B[offB + nPb - 1] < keys_A[offA + nPa - 1]) {
                nPa = cq_upper_bound(keys_A, offA, offA + nPa, keys_B[offB + nPb - 1]) - offA;
            }

            partition_sizes.emplace_back(nPa, nPb);
            offA += nPa, offB += nPb;
        }
    }

    if (offA == nA || offB == nB) {
        if (offA == nA && offB < nB) partition_sizes.emplace_back(0, nB - offB);
        else if (offB == nB && offA < nA) partition_sizes.emplace_back(nA - offA, 0);
    }
    TIMER_LOCAL_STOP;
}

template <typename Key>
static void cq_get_partitions(CQueue<Key>& keys, const size_t M, std::vector<size_t>& partition_sizes) {
    TIMER_LOCAL_START;
    const size_t n = keys.size();
    size_t off = 0;
    while (off < n) {
        if (n - off <= M) {
            partition_sizes.push_back(n - off);
            break;
        }
        else {
            size_t np = MIN(n - off, M);
            Key key = keys[off+np-1];
            np = cq_lower_bound(keys, off, off + np, key) - off;
            partition_sizes.push_back(np);
            off += np;
        }
    }
    TIMER_LOCAL_STOP;
}

template <typename Key>
static void cq_get_search_partitions(CQueue<Key>& keys_A, CQueue<Key>& keys_B, const size_t M,
                              std::vector<std::pair<size_t, size_t>>& partition_sizes) {
    TIMER_LOCAL_START;
    const size_t nA = keys_A.size(), nB = keys_B.size();
    size_t offA = 0, offB = 0;

    while (offA < nA && offB < nB) {
        size_t nPa = MIN(nA - offA, M / 2), nPb = MIN(nB - offB, M / 2);

        if (nPa < M / 2 && nPb < M / 2) {
            partition_sizes.emplace_back(nPa, nPb);
            offA += nPa, offB += nPb;
            break;
        }

        else {
            Key m = MIN(keys_A[offA + nPa - 1], keys_B[offB + nPb - 1]);
            nPa = cq_lower_bound(keys_A, offA, offA + nPa, m) - offA;
            nPb = cq_lower_bound(keys_B, offB, offB + nPb, m) - offB;
            partition_sizes.emplace_back(nPa, nPb);
            offA += nPa, offB += nPb;
        }
    }

    if (offA == nA || offB == nB) {
        if (offA == nA && offB < nB) partition_sizes.emplace_back(0, nB - offB);
        else if (offB == nB && offA < nA) partition_sizes.emplace_back(nA - offA, 0);
    }
    TIMER_LOCAL_STOP;
}

template<typename Key, typename Value>
static void cq_merge_by_key(CQueue<Key> *keys_A, CQueue<Value> *values_A, CQueue<Key> *keys_B, CQueue<Value> *values_B,
                     const size_t M, void* d_buf, CQueue<Key> *keys_C, CQueue<Value> *values_C)
{
    expect(keys_A->size() == values_A->size());
    expect(keys_B->size() == values_B->size());
    expect(keys_C->size() == 0);
    expect(values_C->size() == 0);

    /// find out the partitions
    std::vector<std::pair<size_t, size_t>> partition_sizes;
    cq_get_merge_partitions(*keys_A, *keys_B, M, partition_sizes);

    /// merge partitions
    auto d_keys = (Key*)d_buf;
    auto d_values = (Value*)(d_keys + M);
    for (auto ps : partition_sizes) {
        size_t nA = ps.first, nB = ps.second, rc;
        if (nA && nB) {
            rc = keys_A->pop_front(d_keys, nA); expect(rc == nA);
            rc = keys_B->pop_front(d_keys + nA, nB); expect(rc == nB);
            rc = values_A->pop_front(d_values, nA); expect(rc == nA);
            rc = values_B->pop_front(d_values + nA, nB); expect(rc == nB);
            TIMER_LOCAL_START;
            thrust::sort_by_key(thrust::device, d_keys, d_keys + nA + nB, d_values);
            TIMER_LOCAL_STOP_W_LABEL("Thrust sort in cq_merge_by_key");
            keys_C->push_back(d_keys, nA + nB);
            values_C->push_back(d_values, nA + nB);
        }
        else if (nA) {
            rc = keys_A->pop_front(keys_C, nA); expect(rc == nA);
            rc = values_A->pop_front(values_C, nA); expect(rc == nA);
        }
        else {
            rc = keys_B->pop_front(keys_C, nB); expect(rc == nB);
            rc = values_B->pop_front(values_C, nB); expect(rc == nB);
        }
    }
}

template<typename Key, typename Value>
static void cq_sort_by_key(CQueue<Key> **keys_p, CQueue<Value> **values_p, const size_t M, void* d_buf) {
    TIMER_LOCAL_START;
    auto keys = *keys_p;
    auto values = *values_p;
    const size_t N = keys->size();
    expect(N == values->size());

    auto d_keys = (Key*)d_buf;
    auto d_values = (Value*)(d_keys + M);

    if (N <= M) {
        size_t nk = keys->pop_front(d_keys, N);
        size_t nv = values->pop_front(d_values, N);
        expect(nv == nk);
        expect(nk == N);
        thrust::sort_by_key(thrust::device, d_keys, d_keys + nk, d_values);

        expect(keys->size() == 0);
        expect(values->size() == 0);

        keys->push_back(d_keys, N);
        values->push_back(d_values, N);
    }

    else {
        /// sort key-value pairs in chunks of M
        std::queue<std::pair<CQueue<Key>*, CQueue<Value>*>> sorted;
        expect(sorted.size() == 0);
        while (true) {
            size_t nr = keys->pop_front(d_keys, M);
            size_t nv = values->pop_front(d_values, M);
            expect(nv == nr);
            if (!nr) break;
            thrust::sort_by_key(thrust::device, d_keys, d_keys + nr, d_values);
            auto sorted_keys = new CQueue<Key>();
            auto sorted_values = new CQueue<Value>();
            sorted_keys->push_back(d_keys, nr);
            sorted_values->push_back(d_values, nr);
            sorted.emplace(sorted_keys, sorted_values);
        }

        delete keys;
        delete values;

        /// keep merging until we have 1 block left
        while (sorted.size() > 1) {
            auto kv_pair_left = sorted.front();
            sorted.pop();
            auto kv_pair_right = sorted.front();
            sorted.pop();
            auto out_keys = new CQueue<Key>();
            auto out_values = new CQueue<Value>();
            cq_merge_by_key(kv_pair_left.first, kv_pair_left.second, kv_pair_right.first, kv_pair_right.second,
                            M, d_buf, out_keys, out_values);
            delete kv_pair_left.first; delete kv_pair_left.second;
            delete kv_pair_right.first; delete kv_pair_right.second,
            sorted.emplace(out_keys, out_values);
        }

        auto kv_pair = sorted.front();
        sorted.pop();
        *keys_p = kv_pair.first;
        *values_p = kv_pair.second;
        expect((*keys_p)->size() == N);
        expect((*values_p)->size() == N);
    }
    TIMER_LOCAL_STOP;
}

template <typename Key>
static void cq_count_unique(CQueue<Key> *keys, const size_t M, void* d_buf, CQueue<u4> *counts) {
    std::vector<size_t> partition_sizes;
    cq_get_partitions(*keys, M/2, partition_sizes);
    auto d_keys_in = (Key*)d_buf;
    auto d_keys_out =d_keys_in + M/2;
    auto d_counts = (u4*)(d_keys_in + M);

    size_t off = 0;
    thrust::equal_to<Key> binary_pred;
    thrust::plus<u4> binary_op;
    for (auto np : partition_sizes) {
        keys->copyTo(d_keys_in, off, np);
        auto new_end = thrust::reduce_by_key(thrust::device, d_keys_in, d_keys_in + np,
                                             thrust::make_constant_iterator(1), d_keys_out, d_counts,
                                             binary_pred, binary_op);
        size_t n1 = new_end.first - d_keys_out;
        counts->push_back(d_counts, n1);
        off += np;
    }
}

template <typename Key, typename Value>
static void cq_unique_by_key(CQueue<Key> **keys_p, CQueue<Value> **values_p, const size_t M, void* d_buf) {
    TIMER_LOCAL_START;
    auto keys = *keys_p;
    auto values = *values_p;

    std::vector<size_t> partition_sizes;
    cq_get_partitions(*keys, M, partition_sizes);

    auto tmp_keys = new CQueue<Key>();
    auto tmp_values = new CQueue<Value>();
    auto d_keys = (Key*)d_buf;
    auto d_values = (Value*)(d_keys + M);

    size_t rc;
    for (auto np : partition_sizes) {
        rc = keys->pop_front(d_keys, np); expect(rc == np);
        rc = values->pop_front(d_values, np); expect(rc == np);
        auto new_end = thrust::unique_by_key(thrust::device, d_keys, d_keys + np, d_values);
        size_t new_r = new_end.first - d_keys;

        tmp_keys->push_back(d_keys, new_r);
        tmp_values->push_back(d_values, new_r);
    }

    expect(keys->size() == 0);
    expect(values->size() == 0);

    delete keys;
    delete values;

    *keys_p = tmp_keys;
    *values_p = tmp_values;

    TIMER_LOCAL_STOP;
}

#endif //ASSEMBLY_V3_CQUTILS_H
