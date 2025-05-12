#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


uint32_t arm_sqrt_s32(uint32_t x) {
    if (x == 0) return 0;
    uint32_t res = x;
    uint32_t prev;

    do {
        prev = res;
        res = (res + x / res) / 2;
    } while (res < prev); // Converges when res stops decreasing

    return res;
}