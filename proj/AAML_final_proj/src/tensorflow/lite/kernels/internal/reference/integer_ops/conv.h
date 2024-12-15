/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include <stdio.h>
#include "cfu.h"
// #include "perf.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // perf_enable_counter(6);
  // Define array dimensions
  constexpr int MAX_IM2COL_HEIGHT = 4096;
  constexpr int MAX_IM2COL_WIDTH = 4096;
  constexpr int MAX_KERNEL_HEIGHT = 3600;
  constexpr int MAX_KERNEL_WIDTH = 1024;
  constexpr int MAX_RESULT_HEIGHT = 1024;
  constexpr int MAX_RESULT_WIDTH = 1024;

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;


  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);


  const int M = output_height * output_width;
  // const int K = filter_height * filter_width * filter_input_depth;
  const int N = output_depth;

  const int im2col_height = M;//output_height * output_width;
  const int im2col_width = filter_height * filter_width * input_depth;
  const int kernel_height = im2col_width;
  const int kernel_width = N;//output_depth;


  // Declare arrays
  int8_t im2col[MAX_IM2COL_HEIGHT][MAX_IM2COL_WIDTH];
  int8_t kernel[MAX_KERNEL_HEIGHT][MAX_KERNEL_WIDTH];
  int32_t results[MAX_RESULT_HEIGHT][MAX_RESULT_WIDTH];

  // Add checks to ensure the dimensions are within bounds
  TFLITE_DCHECK(im2col_height <= MAX_IM2COL_HEIGHT);
  TFLITE_DCHECK(im2col_width <= MAX_IM2COL_WIDTH);
  TFLITE_DCHECK(kernel_height <= MAX_KERNEL_HEIGHT);
  TFLITE_DCHECK(kernel_width <= MAX_KERNEL_WIDTH);
  TFLITE_DCHECK(im2col_height <= MAX_RESULT_HEIGHT);
  TFLITE_DCHECK(kernel_width <= MAX_RESULT_WIDTH);

  for (int batch = 0; batch < batches; ++batch) { // batch is 1 here

    // Build the kernel matrix from the filter data
    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        // Loop over each input channel
        for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
            // Precompute the base offset for the current input channel

            // Loop over the filter's height
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                // Precompute the base offset for the current filter row

                // Loop over the filter's width
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    // Calculate the row and column indices for the
                    int kernel_row =in_channel * filter_height * filter_width + filter_y * filter_width + filter_x;
                    int kernel_col = out_channel;

                    // Assign the corresponding value from filter_data to kernel matrix
                    int filter_index = ((out_channel * filter_height + filter_y) * filter_width + filter_x) * filter_input_depth + in_channel;
                    int8_t filter_val = filter_data[filter_index];
                    kernel[kernel_row][kernel_col] = filter_val;
                }
            }
        }
    }

    // Build im2col matrix
    for (int out_y = 0; out_y < output_height; ++out_y) {
        // Calculate the starting y-coordinate in the input image
        const int in_y_origin = (out_y * stride_height) - pad_height;

        // Iterate over the output image width
        for (int out_x = 0; out_x < output_width; ++out_x) {
            // Calculate the starting x-coordinate in the input image
            const int in_x_origin = (out_x * stride_width) - pad_width;

            // Calculate the corresponding row in the im2col matrix
            int im2col_row = out_y * output_width + out_x;

            // Iterate over each input channel
            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {

                // Iterate over the filter height dimension
                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                    const int in_y = in_y_origin + dilation_height_factor * filter_y;

                    // Iterate over the filter width dimension
                    for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                        const int in_x = in_x_origin + dilation_width_factor * filter_x;

                        // Check if the current point is inside the image boundaries
                        const bool is_point_inside_image =
                            (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);

                        // Calculate the corresponding column in the im2col matrix
                        int im2col_col = in_channel * filter_height * filter_width + filter_y * filter_width + filter_x;

                        if (is_point_inside_image) {
                            // Retrieve the input value for valid points
                            int input_index = ((in_y * input_width + in_x) * input_depth + in_channel);
                            int8_t input_val = input_data[input_index];
                            im2col[im2col_row][im2col_col] = input_val;
                        } else {
                            // Apply zero padding for out-of-bounds points
                            im2col[im2col_row][im2col_col] = -input_offset;
                        }
                    }
                }
            }
        }
    }

    cfu_op0(6, input_offset, 0);

    for (int i = 0; i < im2col_height; ++i){
      for (int j = 0; j < kernel_width; ++j){
        results[i][j] = 0;
      }
    }

    // Calculate result matrix with tiling
    constexpr int TILE_SIZE = 32;
    const uint8_t K_t = TILE_SIZE, M_t = TILE_SIZE, N_t = TILE_SIZE; // according to tiling size to change KMN_in
    constexpr int SYSTOLIC_ARR_SIZE = 4;

    uint32_t K_M_N = ((uint32_t) K_t << 16) | ((uint32_t) M_t << 8) | (uint32_t) N_t;
    cfu_op0(1, K_M_N, 0);

    for (int m = 0; m < im2col_height; m += TILE_SIZE) {
      int m_tile = std::min(m + TILE_SIZE, im2col_height) - m;
      for (int n = 0; n < kernel_width; n += TILE_SIZE) {
        int n_tile = std::min(n + TILE_SIZE, kernel_width) - n;
        int32_t C_tile[TILE_SIZE][TILE_SIZE] = {0};
        for (int k = 0; k < im2col_width; k += TILE_SIZE) {

          // initialize systolic array index
          cfu_op0(0, 0, 0);

          int k_tile = std::min(k + TILE_SIZE, im2col_width) - k;
          int8_t A_tile_t[TILE_SIZE][TILE_SIZE];
          int8_t B_tile[TILE_SIZE][TILE_SIZE] = {0};

          for (int i = 0; i < TILE_SIZE; ++i){
            for (int j = 0; j < TILE_SIZE; ++j) {
              A_tile_t[i][j] = -input_offset;
            }
          }

          for (int A_row_idx = 0; A_row_idx < m_tile; A_row_idx++) {
              for (int A_col_idx = 0; A_col_idx < k_tile; A_col_idx++) {
                  A_tile_t[A_col_idx][A_row_idx] = im2col[m + A_row_idx][k + A_col_idx];
              }
          }

          for (int B_col_idx = 0; B_col_idx < k_tile; B_col_idx++) {
              for (int B_row_idx = 0; B_row_idx < n_tile; B_row_idx++) {
                  B_tile[B_col_idx][B_row_idx] = kernel[k + B_col_idx][n + B_row_idx];
              }
          }


          // Load A_tile into CFU
          for (int A_col_offset = 0; A_col_offset < TILE_SIZE; A_col_offset += SYSTOLIC_ARR_SIZE) {
              for (int A_row_index = 0; A_row_index < TILE_SIZE; ++A_row_index) {
                  uint32_t A_tile_value = 0;

                  for (int systolic_idx = 0; systolic_idx < SYSTOLIC_ARR_SIZE; ++systolic_idx) {
                      if ((A_col_offset + systolic_idx) < TILE_SIZE) {
                          int8_t A_value = A_tile_t[A_row_index][A_col_offset + systolic_idx];
                          uint32_t shifted_A_value = static_cast<uint32_t>(static_cast<uint8_t>(A_value))
                                                    << (8 * (SYSTOLIC_ARR_SIZE - 1 - systolic_idx));
                          A_tile_value |= shifted_A_value;
                      }
                  }
                  cfu_op0(2, A_tile_value, 0);
              }
          }

          // Load B_tile into CFU
          for (int B_col_offset = 0; B_col_offset < TILE_SIZE; B_col_offset += SYSTOLIC_ARR_SIZE) {
              for (int B_row_index = 0; B_row_index < TILE_SIZE; ++B_row_index) {
                  uint32_t B_tile_value = 0;

                  for (int systolic_idx = 0; systolic_idx < SYSTOLIC_ARR_SIZE; ++systolic_idx) {
                      if ((B_col_offset + systolic_idx) < TILE_SIZE) {
                          int8_t B_value = B_tile[B_row_index][B_col_offset + systolic_idx];
                          uint32_t shifted_B_value = static_cast<uint32_t>(static_cast<uint8_t>(B_value))
                                                    << (8 * (SYSTOLIC_ARR_SIZE - 1 - systolic_idx));
                          B_tile_value |= shifted_B_value;
                      }
                  }
                  cfu_op0(3, 0, B_tile_value);
              }
          }

          cfu_op0(7, 0, 0); // set in_valid


          while(cfu_op0(4, 0, 0)) {}

          // process output of systolic array to C_tile
          int Tdiv4 = TILE_SIZE / 4;
          int cur_n = 0;
          while (cur_n < Tdiv4) {
            for (int i = 0; i < TILE_SIZE ; i++) {
              for (int j = 0; j < 4 ; j++) {
                C_tile[i][4 * cur_n + j] = cfu_op0(5, j, 0);
              }
            }
            cur_n++;
          }

          for (int mm = 0; mm < m_tile; ++mm) {
            for (int nn = 0; nn < n_tile; ++nn) {
              results[m + mm][n + nn] += C_tile[mm][nn];
            }
          }
        }
      }
    }

    // rebuild output
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = results[out_y * output_width + out_x][out_channel];

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }

  // perf_disable_counter(6);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
