//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_PRELUDE_H
#define ASSEMBLY_V3_PRELUDE_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <stdexcept>
#include <list>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define RESET "\x1B[0m"

#define HIGH BLU
#define MED  MAG
#define LOW  CYN
#define LOGLVL LOW

static char *time_str(){
    time_t rawtime;
    struct tm * timeinfo;
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    static char buf[9];
    strftime (buf, 9,"%T",timeinfo);
    return buf;
}

#define LOG_ERR(fmt, ...) do { \
    fprintf(stderr, "[%s]" RED "ERROR: " fmt RESET " at %s:%i\n",	\
	    time_str(), ##__VA_ARGS__, __FILE__, __LINE__);	\
    assert(false);									\
} while(0)

#define LOG_INFO(fmt, ...) do { \
    fprintf(stdout, "[%s]" GRN "INFO: " fmt RESET "\n", time_str(), ##__VA_ARGS__); \
} while(0)

#define LOG_DEBUG(lvl, fmt, ...) do { if (lvl[3] <= LOGLVL[3]) { \
    fprintf(stdout, "[%s]" lvl "INFO: " fmt RESET "\n", time_str(), ##__VA_ARGS__); \
}} while(0)

#define LOG_WARN(fmt, ...) do { \
    fprintf(stderr, "[%s]" YEL "ERROR: " fmt RESET " at %s:%i\n",	\
	    time_str(), ##__VA_ARGS__, __FILE__, __LINE__);	\
} while(0)

#define CUDA_SAFE(fncall) do { \
    cudaError_t status = (fncall); \
    if (status != cudaSuccess) { \
        LOG_ERR("Call to %s failed because %s (%d)", #fncall, \
            cudaGetErrorString(status), status);\
        exit(1);                         \
    }\
} while(0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define alignup(n, a) (((n) + (a)-1) & ~((a)-1))

#define expect(expression) if (!(expression)) LOG_ERR("Expected " #expression "")

/// aliases and typedefs
#define hvec thrust::host_vector
#define dvec thrust::device_vector
typedef uint8_t u1;
typedef uint32_t u4;
typedef uint64_t u8;

typedef u8 KeyT;
typedef u4 ValT;

/** memory allocation/deallcoation utils */

#define KiB <<10u
#define MiB <<20u
#define GiB <<30u

#define PLEASE_REMOVE

#endif //ASSEMBLY_V3_PRELUDE_H
