/*
 * SPDX-FileCopyrightText: Copyright 2010-2022, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static inline int validate(int8_t *act, const int8_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}

static inline int validate_s16(int16_t *act, const int16_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;
        if (act[i] != ref[i])
        {
            count++;
            printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}

int32_t arm_nn_dequantize(const int8_t val, const int32_t multiplier, const int32_t shift)
{
    int64_t scaled_val = (int64_t)val * (1LL << (31 - shift)); // Undo the shift in requantization
    scaled_val = scaled_val / (int64_t)multiplier; // Undo the multiplier scaling

    return (int32_t)scaled_val;
}


static inline int validate_dynamic(int8_t *act, const int8_t *ref, int size, 
                                   int32_t ref_multiplier, int32_t ref_shift, 
                                   int32_t est_multiplier, int32_t est_shift,
                                   int32_t ref_out_offset, int32_t est_out_offset
                                )
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for (int i = 0; i < size; ++i)
    {
        total++;

        // Reverse requantize the reference and actual values to 32-bit
        int32_t ref_32bit = arm_nn_dequantize(ref[i]-ref_out_offset, ref_multiplier, ref_shift);
        int32_t act_32bit = arm_nn_dequantize(act[i]-est_out_offset, est_multiplier, est_shift);

        printf("  Raw values %d: Act: %d, Ref: %d\n",i, act[i], ref[i]);
        printf("  Scaled values %d: Act: %ld , Ref: %ld\n",i, act_32bit, ref_32bit);

        // Compare the reconstructed 32-bit values
        if (act_32bit != ref_32bit)
        {
            count++;
            printf("ERROR at pos %d: Act: %d (32-bit: %ld) Ref: %d (32-bit: %ld)\r\n", 
                   i, act[i], act_32bit, ref[i], ref_32bit);
            test_passed = false;
        }
    }

    if (!test_passed)
    {
        printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}