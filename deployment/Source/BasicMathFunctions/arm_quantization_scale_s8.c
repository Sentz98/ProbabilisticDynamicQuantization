/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_minimum_s8
 * Description:  Minimum and Maximum
 *
 * $Date:        11 February 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <stdlib.h>
/**
 *  @ingroup Public
 */

/**
 * @addtogroup dynamicQuantization
 * @{
 */

void arm_quantization_scale_s8(const int32_t range, int32_t *multiplier, int32_t *shift) {
    
    *multiplier = (255LL << 31) / range; // Compute scale in Q31 format
    
    // Normalize scale_q31 to ensure the leading bit is in the 2^30 position
    while (*multiplier < (1 << 30)) {
        *multiplier <<= 1;
        (*shift)--;
    }
}

void arm_quantization_scale_s8_v2(
                            const int32_t min, 
                            const int32_t max, 
                            int32_t *multiplier, 
                            int32_t *shift, 
                            int32_t *zero_point) {

    int32_t range = abs(max - min); // Ensure range > 0 (add abs if needed)
    // Compute scale in Q31 format: scale = 255 / range
    *multiplier = (255LL << 31) / range;

    *shift = 0;

    // Normalize multiplier to ensure its leading bit is at position 2^30.
    // Adjust the exponent (shift) accordingly.
    while (*multiplier < (1 << 30)) {
    *multiplier <<= 1;
    (*shift)--;
    }

    // Compute zero_point = round(-min * scale)
    // where real scale = multiplier * 2^(shift) / 2^31.
    // Equivalently:
    //   zero_point = round((-min * multiplier) / 2^(31 - shift))
    int exponent = 31 - *shift;
    int64_t prod = (int64_t)(-min) * (*multiplier);
    *zero_point = (int32_t)((prod + (1LL << (exponent - 1))) >> exponent);
}
