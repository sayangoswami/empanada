//
// Created by sayan on 20-09-2022.
//

#ifndef ASSEMBLY_V3_SINGLETON_H
#define ASSEMBLY_V3_SINGLETON_H

#include "prelude.h"

template <typename C, typename ...Args>
class Singleton {
    Singleton() = default;
    static C* instance;

public:
    ~Singleton() {
        delete instance;
        instance = nullptr;
    }
    static void createInstance(Args...args) {
        expect(instance == nullptr);
        instance = new C(args...);
    }
    static C* getInstance() {
        expect(instance);
        return instance;
    }
};

template <typename C, typename ...Args>
C* Singleton<C, Args...>::instance = nullptr;


template < template<typename T> class C, typename Ta, typename ...Args>
class Singleton1 {
    Singleton1() = default;
    static C<Ta>* instance;

public:
    ~Singleton1() {
        delete instance;
        instance = nullptr;
    }
    static void createInstance(Args...args) {
        expect(instance == nullptr);
        instance = new C<Ta>(args...);
    }
    static C<Ta>* getInstance() {
        expect(instance);
        return instance;
    }
};

template < template<typename T> class C, typename Ta, typename ...Args>
C<Ta>* Singleton1<C, Ta, Args...>::instance = nullptr;

#endif //ASSEMBLY_V3_SINGLETON_H
