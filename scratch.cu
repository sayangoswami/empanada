//
// Created by sayan on 20-09-2022.
//

#include "assembly.cuh"

template <typename T>
struct Pair {
    T x, y;
    Pair(T x, T y) : x(x), y(y) {}
};

void test_singleton() {
    struct S {
        int x, y, z;
        S(int x, int y, int z): x(x), y(y), z(z) {}
    };

    typedef Singleton<S, int, int, int> SS;
    SS::createInstance(1, 2, 3);
    auto sp = SS::getInstance();
    printf("%d %d %d\n", sp->x, sp->y, sp->z);

    auto sp1 = SS::getInstance();
    printf("%d %d %d\n", sp1->x, sp1->y, sp1->z);

    typedef Singleton1<Pair, int, int, int> SS1i;
    SS1i::createInstance(5, 10);
    auto xi1 = SS1i::getInstance();
    printf("%d %d\n", xi1->x, xi1->y);
    auto xi2 = SS1i::getInstance();
    printf("%d %d\n", xi2->x, xi2->y);

    typedef Singleton1<Pair, float, float, float> SS1f;
    SS1f::createInstance(3.14f, 6.78f);
    auto xf1 = SS1f::getInstance();
    printf("%f %f\n", xf1->x, xf1->y);
    auto xf2 = SS1f::getInstance();
    printf("%f %f\n", xf2->x, xf2->y);

}

void test(int argc, char *argv[]) {
    test_singleton();
}