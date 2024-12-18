// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "Systolic_Array/global_buffer_bram.v"
`include "Systolic_Array/TPU.v"

module Cfu (
  input               cmd_valid,
  output reg          cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
  // ===================================================
  // ----------------- STATE PARAMETER -----------------
  // ===================================================
  // CFU FSM state
  reg [1:0] state;
  localparam START_CFU = 2'd0;
  localparam WAIT = 2'd1;

  // MAIN_FUNCT3 : main funcion ID 3 bits
  wire [2:0] MAIN_FUNCT3;
  assign MAIN_FUNCT3 = cmd_payload_function_id[2:0];
  localparam SYSTOLIC_ARRAY = 3'd0;
  localparam SIMD_FC = 3'd1;
  localparam ADD_OP = 3'd2;

  // SUB_FUNCT7 : sub function ID 7 bits
  wire [6:0] SUB_FUNCT7;
  assign SUB_FUNCT7 = cmd_payload_function_id[9:3];
  // For SYSTOLIC ARRAY
  localparam FUNCT_RESET        = 7'd0;
  localparam FUNCT_SET_KMN      = 7'd1;
  localparam FUNCT_STORE_A      = 7'd2;
  localparam FUNCT_STORE_B      = 7'd3;
  localparam FUNCT_IS_BUSY      = 7'd4;
  localparam FUNCT_OUTPUT_C     = 7'd5;
  localparam FUNCT_SET_OFFSET   = 7'd6;
  localparam FUNCT_SET_VALID    = 7'd7;
  // For FC SIMD
  localparam FUNCT_INIT_FC    = 7'd0;
  localparam FUNCT_RST_FC    = 7'd1;
  // For ADD_OP
  // MultiplyByQuantizedMultiplierSmallerThanOneExp
  localparam FUNCT_SET_SRDHM    = 7'd0;
  localparam FUNCT_GET_SRDHM    = 7'd1;
  localparam FUNCT_GET_RDBPOT   = 7'd2;
  localparam FUNCT_CUTOFF       = 7'd3;

  // ===================================================
  // ----------------- OP PARAMETER -----------------
  // ===================================================
  // For SYSTOLIC ARRAY
  reg             in_valid;
  reg [7:0]       K, M, N;
  wire            busy;

  wire A_wr_en, B_wr_en, C_wr_en, C_wr_en_TPU;
  reg  A_wr_en_CFU, B_wr_en_CFU, C_wr_en_CFU;
  wire [2:0] C_write_cnt;

  wire [11:0] A_idx, A_idx_forR, B_idx, B_idx_forR, C_idx, C_idx_forW;
  reg [11:0] A_idx_forW, B_idx_forW, C_idx_forR;

  reg [31:0] A_data_i, B_data_i;
  wire [31:0] A_data_o, B_data_o;
  wire [127:0] C_data_i;
  wire [127:0] C_data_o;
  reg [31:0] offset;

  // For FC SIMD
  reg [15:0] filter_offset, input_offset;
  wire signed [16:0] prod_fc_0, prod_fc_1, prod_fc_2, prod_fc_3;
  wire signed [31:0] sum_prods_fc;

  // For ADD_OP
  reg         overflow;
  reg [63:0]  ab_64;
  wire [31:0] nudge;
  wire [63:0] ab_x2_high32;
  wire [63:0] _ab_x2_high32_;
  wire [31:0] srdhm;

  wire signed [31:0] mask;
  wire signed [31:0] remainder;
  wire signed [31:0] threshold;
  wire signed [31:0] rdbpot;

  wire [31:0] sum;
  wire [31:0] cut_max;
  wire [31:0] cut_min;


  // ==================================================================
  // ------------------ INDEX AND WRITE ENABLE LOGIC ------------------
  // ==================================================================
  assign A_idx = busy ? A_idx_forR : A_idx_forW;
  assign A_wr_en = busy ? 0 : A_wr_en_CFU;
  assign B_idx = busy ? B_idx_forR : B_idx_forW;
  assign B_wr_en = busy ? 0 : B_wr_en_CFU;
  assign C_idx = busy ? C_idx_forW : C_idx_forR;
  assign C_wr_en = busy ? C_wr_en_TPU : C_wr_en_CFU;

  // =======================================================
  // ------------------ SIMD CALCULATION -------------------
  // =======================================================
  assign prod_fc_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[7 : 0]) + $signed(input_offset));
  assign prod_fc_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[15: 8]) + $signed(input_offset));
  assign prod_fc_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[23:16]) + $signed(input_offset));
  assign prod_fc_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(filter_offset))
                    * ($signed(cmd_payload_inputs_1[31:24]) + $signed(input_offset));

  assign sum_prods_fc = prod_fc_0 + prod_fc_1 + prod_fc_2 + prod_fc_3;

  // ===================================================================================
  // ------------------ SATURATING ROUNDING DOUBLING MULTIPLY (SRDHM) ------------------
  // ===================================================================================

  /* SRDHM
    inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                        std::int32_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
    std::int64_t a_64(a);
    std::int64_t b_64(b);
    std::int64_t ab_64 = a_64 * b_64;
    std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    std::int32_t ab_x2_high32 =
        static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
    }
  */

  assign nudge = ab_64[63] ? 32'hc0000001 : 32'h40000000;
  assign ab_x2_high32 = $signed(ab_64) + $signed(nudge);
  assign _ab_x2_high32_ = ab_x2_high32[63] ? -(-ab_x2_high32 >> 31) : ab_x2_high32 >> 31;
  assign srdhm = overflow ? 32'h7fffffff : _ab_x2_high32_;


  // =====================================================================
  // ------------------ ROUNDING DIVIDE BY POT (RDBPOT) ------------------
  // =====================================================================

  /* RDBPOT
    inline IntegerType RoundingDivideByPOT(IntegerType x, int exponent) {
      assert(exponent >= 0);
      assert(exponent <= 31);
      const IntegerType mask = Dup<IntegerType>((1ll << exponent) - 1);
      const IntegerType zero = Dup<IntegerType>(0);
      const IntegerType one = Dup<IntegerType>(1);
      const IntegerType remainder = BitAnd(x, mask);
      const IntegerType threshold =
          Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
      return Add(ShiftRight(x, exponent),
                BitAnd(MaskIfGreaterThan(remainder, threshold), one));
    }
  */

  assign mask = (1 << cmd_payload_inputs_1) - 1;
  assign remainder = cmd_payload_inputs_0 & mask;
  assign threshold = (mask >>> 1) + cmd_payload_inputs_0[31];
  assign rdbpot = $signed($signed(cmd_payload_inputs_0) >>> cmd_payload_inputs_1) +
                  ($signed(remainder) > $signed(threshold));

  // =====================================================
  // ------------------ CUT-OFF MIN/MAX ------------------
  // =====================================================

  /* cutoff_minmax
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                std::max(params.quantized_activation_min, raw_output));
  */

  assign sum = cmd_payload_inputs_0 + cmd_payload_inputs_1;
  assign cut_max = $signed(sum) > $signed(-128) ? sum : $signed(-128);
  assign cut_min = $signed(cut_max) < $signed(127) ? cut_max : $signed(127);


  // =========================================================
  // ------------------ STATE MACHINE LOGIC ------------------
  // =========================================================

  always @(posedge clk) begin
    if (in_valid) begin
      in_valid <= 0;
    end
    if (reset) begin
      in_valid <= 1'b0;
      cmd_ready <= 1'b1;
      rsp_valid <= 1'b0;
      A_idx_forW <= 12'b0;
      B_idx_forW <= 12'd0;
      A_wr_en_CFU <= 1'b0;
      B_wr_en_CFU <= 1'b0;
      offset <= 0;
      state <= START_CFU;
    end else begin
        case (state)
          START_CFU: begin
            if (cmd_valid && cmd_ready) begin
              cmd_ready <= 1'b0;
              rsp_valid <= 1'b1;
              state <= WAIT;
              case (MAIN_FUNCT3)
                SYSTOLIC_ARRAY: begin
                  case (SUB_FUNCT7)
                    FUNCT_RESET: begin // reset idx
                      A_wr_en_CFU <= 1'b0;
                      B_wr_en_CFU <= 1'b0;
                      A_idx_forW <= -1;
                      B_idx_forW <= -1;
                    end
                    FUNCT_SET_KMN: begin
                      K <= cmd_payload_inputs_0[23:16];
                      M <= cmd_payload_inputs_0[15:8];
                      N <= cmd_payload_inputs_0[7:0];
                    end
                    FUNCT_STORE_A: begin // data -> gbuff A
                      A_wr_en_CFU = 1'b1;
                      A_idx_forW <= A_idx_forW + 1;
                      A_data_i <= cmd_payload_inputs_0[31:0];
                    end
                    FUNCT_STORE_B: begin // data -> gbuff B
                      B_wr_en_CFU = 1'b1;
                      B_idx_forW <= B_idx_forW + 1;
                      B_data_i <= cmd_payload_inputs_1[31:0];
                    end
                    FUNCT_IS_BUSY: begin // check cfu is busy or not
                      if (busy) begin
                        rsp_payload_outputs_0 <= 1;
                      end else begin
                        rsp_payload_outputs_0 <= 0;
                        C_idx_forR <= 0;
                      end
                    end
                    FUNCT_OUTPUT_C: begin
                      C_wr_en_CFU = 1'b0;
                      case (cmd_payload_inputs_0[1:0])
                        2'd0: rsp_payload_outputs_0 <= C_data_o[127:96];
                        2'd1: rsp_payload_outputs_0 <= C_data_o[95:64];
                        2'd2: rsp_payload_outputs_0 <= C_data_o[63:32];
                        2'd3: begin
                          rsp_payload_outputs_0 <= C_data_o[31:0];
                          C_idx_forR <= C_idx_forR + 1;
                        end
                        default: ;
                      endcase
                    end
                    FUNCT_SET_OFFSET: offset <= cmd_payload_inputs_0[31:0];
                    FUNCT_SET_VALID: in_valid <= 1'b1;
                    default: ;
                  endcase
                end
                SIMD_FC: begin
                  case (SUB_FUNCT7)
                    FUNCT_INIT_FC:begin
                        filter_offset <= cmd_payload_inputs_0;
                        input_offset <= cmd_payload_inputs_1;
                        rsp_payload_outputs_0 <= 32'b0;
                    end
                    FUNCT_RST_FC: begin
                        filter_offset <= filter_offset;
                        input_offset <= input_offset;
                        rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods_fc;
                    end
                  endcase
                end
                ADD_OP: begin
                  case (SUB_FUNCT7)
                  FUNCT_SET_SRDHM:begin
                    overflow <= (cmd_payload_inputs_0 == 32'h80000000) && (cmd_payload_inputs_1 == 32'h80000000);
                    ab_64 <= $signed(cmd_payload_inputs_0) * $signed(cmd_payload_inputs_1);
                  end
                  FUNCT_GET_SRDHM: begin
                    rsp_payload_outputs_0 <= srdhm;
                  end
                  FUNCT_GET_RDBPOT: begin
                    rsp_payload_outputs_0 <= rdbpot;
                  end
                  FUNCT_CUTOFF:begin
                    rsp_payload_outputs_0 <= cut_min;
                  end
                  endcase
                end
                default: ;
              endcase
            end
          end
          WAIT: begin
            if (rsp_valid && rsp_ready) begin
              cmd_ready <= 1'b1;
              rsp_valid <= 1'b0;
              state <= START_CFU;
            end
          end
          default: state <= START_CFU;
        endcase
    end
  end

  // ===================================================
  // ----------------- MODULE INSTANCES ----------------
  // ===================================================

  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .idx(A_idx),
    .data_in(A_data_i),
    .data_out(A_data_o)
  );

  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_B(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .idx(B_idx),
    .data_in(B_data_i),
    .data_out(B_data_o)
  );
  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_C(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(C_wr_en),
    .idx(C_idx),
    .data_in(C_data_i),
    .data_out(C_data_o)
  );

  TPU #(
    .ARRAY_SIZE(4)
  ) My_TPU(
    .clk            (clk),
    .reset          (reset),
    .in_valid       (in_valid),
    .K              (K),
    .M              (M),
    .N              (N),
    .busy           (busy),
    .A_wr_en        (),
    .A_idx          (A_idx_forR),
    .A_data_i       (),
    .A_data_o       (A_data_o),
    .B_wr_en        (),
    .B_idx          (B_idx_forR),
    .B_data_i       (),
    .B_data_o       (B_data_o),
    .C_wr_en        (C_wr_en_TPU),
    .C_idx          (C_idx_forW),
    .C_data_i       (C_data_i),
    .C_data_o       (C_data_o),
    .offset         (offset),
    .C_write_cnt    (C_write_cnt)
  );

endmodule
