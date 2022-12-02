//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_CONFIG_H
#define ASSEMBLY_V3_CONFIG_H

#include "prelude.h"

struct Option {
    char shortname = 0;
    std::string longname;
    std::string description;
    std::string defaultValue;
    std::string value;
    bool arg = true;

    bool is_required() const { return !shortname; }
};

static void printUsageAndExit(std::list<Option>& options) {
    printf("Usage: assembly_v3");
    for (Option& o : options) {
        if (o.is_required()) printf(" [%s]", o.description.c_str());
    }
    printf(" [options]\n");
    printf("Options:\n");
    for (Option& o : options) {
        if (!o.is_required()) {
            printf("  -%c or --%s, %s, default = %s\n",
                   o.shortname, o.longname.c_str(), o.description.c_str(), o.defaultValue.c_str());
        }
    }
    exit(0);
}

static void parse_args(int argc, char *argv[], std::list<Option>& options) {
    auto it = options.begin();
    --argc; ++argv;
    while (argc) {
        std::string arg = argv[0];
        if (it->is_required()) {
            it->value = arg;
            ++it, --argc, ++argv;
        }
        else {
            auto it1 = it;
            bool found = false;
            while (it1 != options.end()) {
                if (arg[1] == '-') {
                    LOG_ERR("Not implemented yet. Use short forms instead."); /// fixme
                }
                else if (arg[1] == it1->shortname) {
                    it1->value = argv[1];
                    argc -= 2, argv += 2;
                    found = true;
                }
                ++it1;
            }
            if (!found) {
                printf("Invalid option %s\n", arg.c_str());
                printUsageAndExit(options);
            }
        }
    }
    while (it != options.end()) {
        if (it->is_required()) printUsageAndExit(options);
        else {
            if (it->value.empty()) it->value = it->defaultValue;
            ++it;
        }
    }
}

static inline size_t hmsize2bytes(std::string& hsize) {
    char unit = hsize.back();
    if (unit >= 48 && unit <= 57)
        return std::stoul(hsize);
    else {
        hsize.pop_back();
        size_t size = std::stoul(hsize);
        switch (unit) {
            case 'k':
            case 'K':
                size = size KiB;
                break;
            case 'm':
            case 'M':
                size = size MiB;
                break;
            case 'g':
            case 'G':
                size = size GiB;
                break;
            default:
                LOG_ERR("Unknown unit %c.", unit);
        }
        return size;
    }
}

class Config {
private:
    Config() {}
    ~Config() {}

public:
    std::string input, output;
    int min_ovlp;
    size_t map_mem, reduce_mem, base_blocksz, kv_chunksz;

    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    Config(const Config&) = delete;
    Config& operator = (const Config&) = delete;

    void init(int argc, char *argv[]) {
        Option options[] = {
                { 0, "", "input fastq"},
                { 0, "", "output gfa"},
                { 'o', "min-overlap", "minimum overlap length", "31" },
                { 'm', "map-mem", "device memory during the fingerprint generation phase", "1g" },
                { 'r', "reduce-mem", "device memory during the overlap-detection phase", "1g" },
                { 'b', "base-blocksz", "block size (count) of transposed bases", "32m" },
                { 'c', "kv-chunksz", "block size (count) of key-value pairs", "64m" }
        };
        std::list<Option> optionsList(std::begin(options), std::end(options));
        parse_args(argc, argv, optionsList);
        auto it = optionsList.begin();
        input = (it++)->value;
        output = (it++)->value;
        min_ovlp = hmsize2bytes((it++)->value);
        map_mem = hmsize2bytes((it++)->value);
        reduce_mem = hmsize2bytes((it++)->value);
        base_blocksz = hmsize2bytes((it++)->value);
        kv_chunksz = hmsize2bytes((it++)->value);
    }
};

#endif //ASSEMBLY_V3_CONFIG_H
