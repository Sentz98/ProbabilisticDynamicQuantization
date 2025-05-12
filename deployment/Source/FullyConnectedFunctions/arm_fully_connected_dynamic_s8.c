/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_fully_connected_dynamic_s8
 * Description:  Fully connected function with dynamic Quantization.
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
#include <stdio.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/*
 * S8 dynamically quantized fully-connected and matrix multiplication layer function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */


//TODO SPOSTA STA CAZZO DI FUNZIONE
/**
  @brief         Sum of the elements of an int8_t vector.
  @param[in]     pSrc       points to the input vector
  @param[in]     blockSize  number of samples in input vector
  @param[out]    pResult    sum of the elements value returned here
  
  @details       This function computes the sum of an int8_t vector
                 using a 32-bit accumulator to avoid overflow issues.
*/
void arm_sum_s8(
  const int8_t * pSrc,
        uint32_t blockSize,
        int32_t input_offset,
        int32_t * pResult)
{
    uint32_t blkCnt;    /* Loop counter */
    int32_t sum = 0;    /* Accumulator */
    int8_t in;          /* Temporary input variable */

    /* Loop unrolling: Process four elements at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        in = *pSrc++;
        sum += (int16_t)in + input_offset;

        in = *pSrc++;
        sum += (int16_t)in + input_offset;

        in = *pSrc++;
        sum += (int16_t)in + input_offset;

        in = *pSrc++;
        sum += (int16_t)in + input_offset;

        blkCnt--;
    }

    /* Process remaining elements */
    blkCnt = blockSize % 4U;

    while (blkCnt > 0U)
    {
        in = *pSrc++;
        sum += (int16_t)in + input_offset;

        blkCnt--;
    }

    /* Store result */
    *pResult = sum;
}

arm_cmsis_nn_status arm_fully_connected_dynamic_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const cmsis_nn_filter_stats *filter_stats,
                                           const int8_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output)
{
    int32_t batch_cnt = input_dims->n;

    int8_t imgsize = input_dims->h*input_dims->w*input_dims->c;
    // printf("imgsize: %d\n", imgsize);
    // printf("WEIGHT MEAN: %d\n",filter_stats->mean);
    // printf("WEIGHT STD: %d\n",filter_stats->std);
    // printf("--------------\n");

    cmsis_nn_per_tensor_quant_params quant_params;
    int32_t range = 0;
    const int8_t *pSrc = input;

    const int8_t *weight_mean = filter_stats->mean;
    const int8_t *weight_std = filter_stats->std;

    while (batch_cnt)
    {
        int32_t squareSum = 0;     // Idk if this need to be uint64_t to prevent overflow
        int32_t normSum = 0;

        arm_power_s8(pSrc, imgsize, fc_params->input_offset, &squareSum);
        arm_sum_s8(pSrc, imgsize, fc_params->input_offset, &normSum);
        // printf("B%ld - SQUARED SUM: %ld\n", batch_cnt, squareSum); 
        // printf("B%ld - NORMAL SUM: %ld\n", batch_cnt, normSum);

        int32_t l2_norm = arm_sqrt_s32(squareSum);
        printf("B%ld - L2 NORM: %ld\n", batch_cnt, l2_norm);
        
        int32_t q_min = weight_mean[0] * normSum - 2 * weight_std[0] * l2_norm;
        int32_t q_max = weight_mean[0] * normSum + 2 * weight_std[0] * l2_norm;

        range = (abs(q_max-q_min) > range) ? abs(q_max-q_min) : range;

        // printf("B%ld - ESTIMATED QMAX: %ld\n", batch_cnt, q_max); 
        // printf("B%ld - ESTIMATED QMIN: %ld\n", batch_cnt, q_min);
        // printf("B%ld - ESTIMATED RANGEmax: %ld\n", batch_cnt, range);
        // printf("--------------\n");

        pSrc += filter_dims->n;
        batch_cnt--;
    };

    arm_quantization_scale_s8(range, &quant_params.multiplier, &quant_params.shift);

    // printf("ESTIMATED MULTIPLIER: %ld\n", quant_params.multiplier); 
    // printf("ESTIMATED SHIFT: %ld\n", quant_params.shift);

    return arm_fully_connected_s8(ctx, fc_params, &quant_params, input_dims, input, filter_dims, kernel,
                           bias_dims, bias, output_dims, output);
}

/**
 * @} end of FC group
 */
