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
 * Title:        arm_convolve_dynamic_s8.c
 * Description:  s8 version of convolution using dynamic symmetric quantization.
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
 * @addtogroup NNConv
 * @{
 */

/*
 * S8 convolution function with dynamic quantization.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */

arm_cmsis_nn_status arm_convolve_dynamic_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_dynamic_params *conv_dynamic_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const cmsis_nn_filter_stats *filter_stats,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *upscale_dims,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Check if batch size is not 1
    if (input_dims->n != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const uint16_t input_ch = input_dims->c;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t stride_x = conv_dynamic_params->stride.w;
    const uint16_t stride_y = conv_dynamic_params->stride.h;
    const uint16_t pad_x = conv_dynamic_params->padding.w;
    const uint16_t pad_y = conv_dynamic_params->padding.h;
    const int32_t dilation_x = conv_dynamic_params->dilation.w;
    const int32_t dilation_y = conv_dynamic_params->dilation.h;
    const int32_t sampling_stride = conv_dynamic_params->sampling_stride;

    const int8_t *weight_mean = filter_stats->mean;
    const int8_t *weight_std = filter_stats->std;

    // Allocate q_min and q_max arrays from the context buffer
    int32_t *q_min = (int32_t *)ctx->buf;
    int32_t *q_max = q_min + output_ch;
    // Initialize q_min and q_max arrays to 0
    memset(q_min, 0, 2 * output_ch * sizeof(int32_t));

    // Iterate over each output pixel (looping over patches)
    for (int i_out_y = 0; i_out_y < output_y; i_out_y += sampling_stride)
    {
        for (int i_out_x = 0; i_out_x < output_x; i_out_x += sampling_stride)
        {
            // Compute base indices in the input tensor
            const int32_t base_idx_x = i_out_x * stride_x - pad_x;
            const int32_t base_idx_y = i_out_y * stride_y - pad_y;

            uint32_t squareSum = 0;     // Idk if this need to be uint64_t to prevent overflow
            int32_t normSum = 0;

            for (int i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
            {
                for (int i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                {
                    int32_t k_y = base_idx_y + i_ker_y * dilation_y;
                    int32_t k_x = base_idx_x + i_ker_x * dilation_x;

                    if (k_y >= 0 && k_y < input_dims->h && k_x >= 0 && k_x < input_dims->w)
                    {
                    	int ch = k_y * input_dims->w + k_x;
                    	for (int pixel = ch; pixel < ch + input_ch; pixel ++){
                    		int8_t pixel_val = input_data[pixel];
							int32_t squaredValue = (int32_t)pixel_val * (int32_t)pixel_val; // Use int32_t to avoid overflow
							squareSum += squaredValue;
							normSum += pixel_val;  // Allow negative values
                    	}
                    }
                }
            }

            uint32_t l2_norm = arm_sqrt_s32(squareSum);
            
            // Compute q_min and q_max in fixed-point representation
            for (int i = 0; i < output_ch; i++) {
            
                // Compute new_q_min and new_q_max for this channel
                int32_t new_q_min = weight_mean[i] * normSum - 2 * weight_std[i] * l2_norm;
                int32_t new_q_max = weight_mean[i] * normSum + 2 * weight_std[i] * l2_norm;
            
                // Update q_min and q_max for the channel
                if (q_min[i] == 0) {
                    q_min[i] = new_q_min;
                    q_max[i] = new_q_max;
                } else {
                    q_min[i] = (new_q_min < q_min[i]) ? new_q_min : q_min[i];
                    q_max[i] = (new_q_max > q_max[i]) ? new_q_max : q_max[i];
                }
            }             
        }
    }

    for (int i = 0; i < output_ch; i++) {
        printf("MIN %d: %ld\n", i, q_min[i]);
        printf("MAX %d: %ld\n", i, q_max[i]);

        // Compute per-channel multiplier, shift, and zero_point
        int32_t ch_multiplier, ch_shift, ch_zero_point;
        arm_quantization_scale_s8_v2(q_min[i], q_max[i], &ch_multiplier, &ch_shift, &ch_zero_point);

        // Assign to arrays
        quant_params->multiplier[i] = ch_multiplier;
        quant_params->shift[i] = ch_shift;
        // zero_points[i] = ch_zero_point;  // Optional: use if needed
    }

    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = conv_dynamic_params->input_offset;
    conv_params.output_offset = 0; //zero_point; TODO SISTEMAAAAAAAAAAAAAAAA
    conv_params.stride = conv_dynamic_params->stride;
    conv_params.padding = conv_dynamic_params->padding;
    conv_params.dilation = conv_dynamic_params->dilation;
    conv_params.activation = conv_dynamic_params->activation;
    

    // Call original convolution with computed quant_params
    return arm_convolve_s8(ctx, &conv_params, quant_params, input_dims, input_data, filter_dims, filter_data,
                           bias_dims, bias_data, upscale_dims, output_dims, output_data);
}



/////// TEST PER CHANNEL QUANTIZATION

// arm_cmsis_nn_status arm_convolve_dynamic_perchannel_s8(const cmsis_nn_context *ctx,
//     const cmsis_nn_conv_dynamic_params *conv_dynamic_params,
//     const cmsis_nn_dims *input_dims,
//     const int8_t *input_data,
//     const cmsis_nn_dims *filter_dims,
//     const cmsis_nn_filter_stats *filter_stats,
//     const int8_t *filter_data,
//     const cmsis_nn_dims *bias_dims,
//     const int32_t *bias_data,
//     const cmsis_nn_dims *upscale_dims,
//     const cmsis_nn_dims *output_dims,
//     int8_t *output_data)
// {
//     (void)bias_dims;

//     if (ctx->buf == NULL)
//     {
//     return ARM_CMSIS_NN_ARG_ERROR;
//     }

//     // Check if batch size is not 1
//     if (input_dims->n != 1)
//     {
//     return ARM_CMSIS_NN_ARG_ERROR;
//     }

//     const uint16_t output_x = output_dims->w;
//     const uint16_t output_y = output_dims->h;
//     const uint16_t output_ch = output_dims->c;
//     const uint16_t kernel_x = filter_dims->w;
//     const uint16_t kernel_y = filter_dims->h;
//     const uint16_t stride_x = conv_dynamic_params->stride.w;
//     const uint16_t stride_y = conv_dynamic_params->stride.h;
//     const uint16_t pad_x = conv_dynamic_params->padding.w;
//     const uint16_t pad_y = conv_dynamic_params->padding.h;
//     const int32_t dilation_x = conv_dynamic_params->dilation.w;
//     const int32_t dilation_y = conv_dynamic_params->dilation.h;
//     const int32_t sampling_stride = conv_dynamic_params->sampling_stride;

//     const int16_t *weight_mean = filter_stats->mean;
//     const int16_t *weight_std = filter_stats->std;

//     // Allocate q_min and q_max arrays from the context buffer
//     int32_t *q_min = (int32_t *)ctx->buf;
//     int32_t *q_max = q_min + output_ch;

//     // Initialize q_min and q_max arrays to 0
//     memset(q_min, 0, 2 * output_ch * sizeof(int32_t));

//     // Iterate over each output pixel (looping over patches)
//     for (int i_out_y = 0; i_out_y < output_y; i_out_y += sampling_stride)
//     {
//         for (int i_out_x = 0; i_out_x < output_x; i_out_x += sampling_stride)
//         {
//             // Compute base indices in the input tensor
//             const int32_t base_idx_x = i_out_x * stride_x - pad_x;
//             const int32_t base_idx_y = i_out_y * stride_y - pad_y;

//             uint32_t squareSum = 0;     // Idk if this need to be uint64_t to prevent overflow
//             int32_t normSum = 0;

//             for (int i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
//             {
//                 for (int i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
//                 {
//                     int32_t k_y = base_idx_y + i_ker_y * dilation_y;
//                     int32_t k_x = base_idx_x + i_ker_x * dilation_x;

//                     if (k_y >= 0 && k_y < input_dims->h && k_x >= 0 && k_x < input_dims->w)
//                     {
//                         int8_t pixel_val = input_data[k_y * input_dims->w + k_x];
//                         int32_t squaredValue = (int32_t)pixel_val * (int32_t)pixel_val; // Use int32_t to avoid overflow
//                         squareSum += squaredValue;
//                         normSum += pixel_val;  // Allow negative values
//                     }
//                 }
//             }

//             uint32_t l2_norm = arm_sqrt_s32(squareSum);
            
//             // Compute q_min and q_max in fixed-point representation
//             for (int i = 0; i < output_ch; i++) {
            
//                 // Compute new_q_min and new_q_max for this channel
//                 int32_t new_q_min = weight_mean[i] * normSum - 2 * weight_std[i] * l2_norm;
//                 int32_t new_q_max = weight_mean[i] * normSum + 2 * weight_std[i] * l2_norm;
            
//                 // Update q_min and q_max for the channel
//                 if (q_min[i] == 0) {
//                     q_min[i] = new_q_min;
//                     q_max[i] = new_q_max;
//                 } else {
//                     q_min[i] = (new_q_min < q_min[i]) ? new_q_min : q_min[i];
//                     q_max[i] = (new_q_max > q_max[i]) ? new_q_max : q_max[i];
//                 }
//             }           
//         }
//     }

//     cmsis_nn_per_channel_quant_params quant_params;

//     // Allocate space for multipliers, shifts, and zero_points
//     quant_params.multiplier = (int32_t *)malloc(output_ch * sizeof(int32_t));
//     quant_params.shift = (int32_t *)malloc(output_ch * sizeof(int32_t));
//     int32_t *zero_points = (int32_t *)malloc(output_ch * sizeof(int32_t)); // Optional: if needed

//     if (!quant_params.multiplier || !quant_params.shift || !zero_points) {
//         free(quant_params.multiplier);
//         free(quant_params.shift);
//         free(zero_points);
//         return ARM_CMSIS_NN_ARG_ERROR;
//     }

//     for (int i = 0; i < output_ch; i++) {
//         // Get per-channel q_min and q_max
//         int32_t ch_q_min = q_min[i];
//         int32_t ch_q_max = q_max[i];

//         // Compute per-channel multiplier, shift, and zero_point
//         int32_t ch_multiplier, ch_shift, ch_zero_point;
//         arm_quantization_scale_s8_v2(&ch_q_min, &ch_q_max, &ch_multiplier, &ch_shift, &ch_zero_point);

//         // Assign to arrays
//         quant_params.multiplier[i] = ch_multiplier;
//         quant_params.shift[i] = ch_shift;
//         zero_points[i] = ch_zero_point;  // Optional: use if needed
//     }

//     cmsis_nn_conv_params conv_params;
//     conv_params.input_offset = conv_dynamic_params->input_offset;
//     conv_params.output_offset = zero_points[0]; // TODO CAPIRE CHE CAZZO FARE
//     conv_params.stride = conv_dynamic_params->stride;
//     conv_params.padding = conv_dynamic_params->padding;
//     conv_params.dilation = conv_dynamic_params->dilation;
//     conv_params.activation = conv_dynamic_params->activation;


//     // Call original convolution with computed quant_params
//     return arm_convolve_s8(ctx, &conv_params, &quant_params, input_dims, input_data, filter_dims, filter_data,
//     bias_dims, bias_data, upscale_dims, output_dims, output_data);
// }
