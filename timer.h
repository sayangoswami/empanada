//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_TIMER_H
#define ASSEMBLY_V3_TIMER_H

#include <iostream>
#include <chrono>

struct Timer {
private:
    FILE *fp;
    std::chrono::steady_clock::time_point start;
    Timer() {
        std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm ltime;
        localtime_r(&t, &ltime);
        fp = fopen("timings.csv", "w");
        fprintf(fp, "## %s", asctime(&ltime));
        fprintf(fp, "Phase,Time (us)\n");
    }
    ~Timer() {
        fclose(fp);
    }
public:
    static Timer& getInstance() {
        static Timer instance;
        return instance;
    }

    Timer(const Timer&) = delete;
    Timer& operator = (const Timer&) = delete;

    inline void reset() {
        start = std::chrono::steady_clock::now();
    }

    inline void record(std::chrono::steady_clock::time_point lstart, const char* label) {
        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - lstart;
        auto ct = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        fprintf(fp, "%s,%ld\n", label, ct);
    }

    inline void record(const char* label) {
        record(start, label);
    }
};

#ifdef TIMINGS
#define TIMER_GLOBAL_RESET Timer::getInstance().reset()
#define TIMER_GLOBAL_RECORD(label) Timer::getInstance().record(label)
#define TIMER_LOCAL_START auto _local_timer_start = std::chrono::steady_clock::now()
#define TIMER_LOCAL_STOP Timer::getInstance().record(_local_timer_start, __FUNCTION__)
#define TIMER_LOCAL_STOP_W_LABEL(label) Timer::getInstance().record(_local_timer_start, label)
#else
#define TIMER_GLOBAL_RESET 0
#define TIMER_GLOBAL_RECORD(label) 0
#define TIMER_LOCAL_START 0
#define TIMER_LOCAL_STOP 0
#endif

#endif //ASSEMBLY_V3_TIMER_H
