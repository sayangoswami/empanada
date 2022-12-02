//
// Created by sayan on 20-09-2022.
//

#include "assembly.cuh"

extern void test(int argc, char *argv[]);

int main(int argc, char *argv[]) {
#ifdef TESTING
    test(argc, argv);
#else
    Config& config = Config::getInstance();
    config.init(argc, argv);
    std::ifstream file(config.input, std::ios::binary);
    size_t Mm = config.map_mem / 16;
    Kmp::createInstance(config.kv_chunksz);
    Vmp::createInstance(config.kv_chunksz);

    dvec<char> input, transposed;
    dvec<int> stencil, lengths, offsets;
    dvec<KeyT> fingerprints;
    dvec<ValT> read_ids;
    dvec<u1> fplens;

    CUDA_SAFE(cudaMemcpyToSymbol(d_revc, h_revc, 8));
    precomputeModulos();

    auto basesX = new TBases();
    auto cqfp = new CQueue<KeyT>();
    auto cqrid = new CQueue<ValT>();

    bool done = false;
    u4 read_id_offset = 0;
    int readlen = 0;
    LOG_INFO("Reading file %s", config.input.c_str());

#ifdef TIMINGS
    Timer::getInstance().reset();
#endif

    while (!done) {
        done = read_block(file, Mm, input, stencil, lengths, offsets);
        if (!lengths.empty()) {
            int nrows, ncols;
            transpose_reads(input, lengths, transposed, nrows, ncols);
            if (!readlen) readlen = nrows;
            else expect(nrows == readlen);
            get_reverse_complements(input, stencil, lengths, offsets);
            generate_fingerprint_read_pairs(read_id_offset, input, stencil, fplens, fingerprints, read_ids);
            TIMER_GLOBAL_RESET;
            cqfp->push_back(thrust::raw_pointer_cast(fingerprints.data()), fingerprints.size());
            cqrid->push_back(thrust::raw_pointer_cast(read_ids.data()), read_ids.size());
            basesX->append(transposed, nrows, ncols);
            TIMER_GLOBAL_RECORD("copy_f_b_to_dev");
            read_id_offset += lengths.size();
        }
    }
    LOG_INFO("Map Phase over..");
    input.clear(); dvec<char>().swap(input);
    transposed.clear(); dvec<char>().swap(transposed);
    stencil.clear(); dvec<int>().swap(stencil);
    lengths.clear(); dvec<int>().swap(lengths);
    offsets.clear(); dvec<int>().swap(offsets);
    fplens.clear(); dvec<u1>().swap(fplens);
    fingerprints.clear(); dvec<KeyT>().swap(fingerprints);
    read_ids.clear(); dvec<ValT>().swap(read_ids);

    size_t Mr = config.reduce_mem / 16;
    init_match(Mr);
    build_graph(readlen, cqfp, cqrid, basesX, config.output);

    cudaDeviceReset();
#endif

    return 0;
}