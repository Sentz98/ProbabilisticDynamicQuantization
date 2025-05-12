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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/dynamic/test_data.h"
#include "../Utils/validate.h"

void basic_arm_convolve_dynamic_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DYNAMIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_dynamic_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_filter_stats filter_stats;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = dynamic_biases;
    const int8_t *kernel_data = dynamic_weights;
    const int8_t *input_data = dynamic_input;
    const int8_t *output_ref = dynamic_output_ref;
    const int32_t output_ref_size = DYNAMIC_DST_SIZE;

    input_dims.n = DYNAMIC_INPUT_BATCHES;
    input_dims.w = DYNAMIC_INPUT_W;
    input_dims.h = DYNAMIC_INPUT_H;
    input_dims.c = DYNAMIC_IN_CH;
    filter_dims.w = DYNAMIC_FILTER_X;
    filter_dims.h = DYNAMIC_FILTER_Y;
    filter_dims.c = DYNAMIC_IN_CH;
    output_dims.w = DYNAMIC_OUTPUT_W;
    output_dims.h = DYNAMIC_OUTPUT_H;
    output_dims.c = DYNAMIC_OUT_CH;

    conv_params.padding.w = DYNAMIC_PAD_X;
    conv_params.padding.h = DYNAMIC_PAD_Y;
    conv_params.stride.w = DYNAMIC_STRIDE_X;
    conv_params.stride.h = DYNAMIC_STRIDE_Y;
    conv_params.dilation.w = DYNAMIC_DILATION_X;
    conv_params.dilation.h = DYNAMIC_DILATION_Y;

    conv_params.input_offset = DYNAMIC_INPUT_OFFSET;
    conv_params.activation.min = DYNAMIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DYNAMIC_OUT_ACTIVATION_MAX;

    filter_stats.mean = (int8_t *)dynamic_weights_mean;
    filter_stats.std = (int8_t *)dynamic_weights_std;
    // printf("REFERENCE MULTIPLIER: %ld\n", *(int32_t *)dynamic_output_mult); 
    // printf("REFERENCE SHIFT %ld\n", *(int32_t *)dynamic_output_shift);


    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_per_channel_quant_params quant_params;

    // Allocate space for multipliers and shifts
    quant_params.multiplier = (int32_t *)malloc(output_dims.c  * sizeof(int32_t));
    quant_params.shift = (int32_t *)malloc(output_dims.c  * sizeof(int32_t));

    arm_cmsis_nn_status result = arm_convolve_dynamic_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 &filter_stats,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    // Print multiplier array
    printf("MULTIPLIER:");
    for (int i = 0; i < output_dims.c; i++) {
        printf("%ld", quant_params.multiplier[i]);
        if (i < output_dims.c - 1) printf(",");
    }
    printf("\n");

    // Print shift array
    printf("SHIFT:");
    for (int i = 0; i < output_dims.c; i++) {
        printf("%ld", quant_params.shift[i]);
        if (i < output_dims.c - 1) printf(",");
    }
    printf("\n");

    // Print output array
    printf("OUTPUT:");
    for (int i = 0; i < output_ref_size; i++) {
        printf("%d", output[i]);
        if (i < output_ref_size - 1) printf(",");
    }
    printf("\n");

    fflush(stdout); // Ensure output is flushed

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}
