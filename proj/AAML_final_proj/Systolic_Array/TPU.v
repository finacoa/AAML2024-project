`include "Systolic_Array.v"

module TPU #(parameter ARRAY_SIZE = 4)(
    input               clk,
    input               reset,

    input               in_valid,
    input         [7:0] K,
    input         [7:0] M,
    input         [7:0] N,
    output reg          busy,

    output              A_wr_en,
    output reg   [11:0] A_index,
    output       [31:0] A_data_in,
    input signed [31:0] A_data_out,

    output              B_wr_en,
    output reg   [11:0] B_index,
    output       [31:0] B_data_in,
    input signed [31:0] B_data_out,

    output reg          C_wr_en,
    output reg   [11:0] C_index,
    output reg  [127:0] C_data_in,
    input       [127:0] C_data_out,
    input signed [31:0] offset,
    output reg    [2:0] C_write_cnt

);

reg   [2:0] state;
localparam IDLE = 3'd0;
localparam WORK = 3'd1;
localparam WRITE = 3'd2;
localparam DONE = 3'd3;
localparam ERROR = 3'd4;

reg PE_rst;

assign A_wr_en = 0;
assign B_wr_en = 0;
assign A_data_in = 32'd0;
assign B_data_in = 32'd0;

reg [7:0] Mdiv4, Ndiv4;
reg [7:0] K_in, M_in, N_in;

always @(posedge clk) begin
    if (reset) begin
        Mdiv4 <= 0;
        Ndiv4 <= 0;
        K_in <= 0;
        M_in <= 0;
        N_in <= 0;
    end else if (in_valid) begin
        K_in = K;
        M_in = M;
        N_in = N;
        if (M_in <= 4) begin
            Mdiv4 = 1;
        end else if (M_in[1:0] == 2'b00) begin
            Mdiv4 = M_in/4;
        end else begin
            Mdiv4 = M_in/4 + 1;
        end
        if (N_in <= 4) begin
            Ndiv4 = 1;
        end else if (N_in[1:0] == 2'b00) begin
            Ndiv4 = N_in/4;
        end else begin
            Ndiv4 = N_in/4 + 1;
        end
    end

end

//* Implement your design here
reg signed [8:0] A_in [ARRAY_SIZE-1:0];
reg signed [7:0] B_in [ARRAY_SIZE-1:0];
wire signed [127:0] C_SA_out [ARRAY_SIZE-1:0];


reg signed [31:0] A_reg [ARRAY_SIZE-1:0];
reg signed [31:0] B_reg [ARRAY_SIZE-1:0];
reg signed [127:0] C_reg [ARRAY_SIZE-1:0];

// reg_state control which reg (0, 1, 2, 3) should give data to A_in (0, 1, 2, 3)
reg [1:0] A_reg_state [ARRAY_SIZE-1:0];
reg [1:0] B_reg_state [ARRAY_SIZE-1:0];
reg [2:0] C_reg_state;

reg [7:0] m, k, n;

integer i;
always @(posedge clk) begin
    if (reset) begin
        PE_rst <= 1;
        state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_index <= 0;
        B_index <= 0;
        C_index <= 0;
        C_write_cnt <= 0;
        busy <= 0;

        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            A_in[i] <= 0;
            B_in[i] <= 0;
            A_reg[i] <= 32'd0;
            B_reg[i] <= 32'd0;
            C_reg[i] <= {128{1'b0}};
            A_reg_state[i] <= 0;
            B_reg_state[i] <= 0;
        end
        C_reg_state <= 3'd0;


    end else if (in_valid) begin
        PE_rst <= 1;
        state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_index <= 0;
        B_index <= 0;
        C_index <= 0;
        C_write_cnt <= 0;
        busy <= 1;
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            A_reg[i] <= 32'd0;
            B_reg[i] <= 32'd0;
            C_reg[i] <= {128{1'b0}};
            if (i == 0) begin
                // the first index0 should be 0
                A_reg_state[i] <= 0;
                B_reg_state[i] <= 0;
            end else begin
                // 3,2,1
                A_reg_state[i] <= ARRAY_SIZE - i;
                B_reg_state[i] <= ARRAY_SIZE - i;
            end
        end
        C_reg_state <= 3'd0;

    end else if (busy) begin
        case (state)
            IDLE: begin
                if (busy) begin
                    state <= WORK;
                end else begin
                    state <= IDLE;
                end
            end

            WORK: begin
                PE_rst <= 0;
                C_wr_en <= 0; // switch to WORK: 1 cycle, next cycle: C_wr_en = 0

                A_reg[A_reg_state[0]] = k < K_in? A_data_out : 0;
                B_reg[B_reg_state[0]] = k < K_in? B_data_out : 0;

                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin // start from A_in[1]
                    if (i == 0) begin
                        // specially processing A_in[0]
                        if (k < K_in) begin
                            A_in[0] <= $signed(A_data_out[31:24]) + $signed(offset);
                            B_in[0] <= $signed(B_data_out[31:24]);
                        end else begin
                            A_in[0] <= 0;
                            B_in[0] <= 0;
                        end
                    end else begin
                        if (k < K_in + i) begin
                            A_in[i] <= $signed(A_reg[A_reg_state[i]][(8 * (3 - (i % 4))) + 7 -: 8]) + $signed(offset);
                            B_in[i] <= $signed(B_reg[B_reg_state[i]][(8 * (3 - (i % 4))) + 7 -: 8]) ;
                        end else begin
                            A_in[i] <= 0;
                            B_in[i] <= 0;
                        end
                    end
                end


                for (i = ARRAY_SIZE-1; i > 0; i = i - 1) begin
                    A_reg_state[i] <= A_reg_state[i-1];
                    B_reg_state[i] <= B_reg_state[i-1];
                end
                // the last one move to the first one
                A_reg_state[0] <= A_reg_state[ARRAY_SIZE-1];
                B_reg_state[0] <= B_reg_state[ARRAY_SIZE-1];

                A_index <= m * K_in + k + 1;
                B_index <= n * K_in + k + 1;

                k <= k + 1;

                if (k + 3 > K_in + 6) begin  // will switch to WRITE state at cycle = k + 2, then write at cycle = k + 3 ~ k + 6
                    C_reg_state <= 2'd0;
                    C_wr_en <= 1;
                    state <= WRITE;
                end
            end


            WRITE: begin
                C_index = n * M_in + m * 4 + C_write_cnt;
                C_data_in = C_SA_out[C_write_cnt];

                if (m + 1 == Mdiv4) begin // m stuck at 3, n stuck at 0
                    if (m * 4 + C_write_cnt >= M_in - 1) begin // 3 * 4 + () >= 15

                        if (n + 1> Ndiv4) begin
                            state <= DONE;
                        end else begin
                            PE_rst <= 1;
                            k <= 0;
                            m <= 0;
                            n <= n + 1;
                            A_index <= 0;
                            B_index <= (n + 1) * K_in;
                            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                                A_in[i] <= 0;
                                B_in[i] <= 0;
                                A_reg[i] <= 32'd0;
                                B_reg[i] <= 32'd0;
                                C_reg[i] <= {128{1'b0}};
                                if (i == 0) begin
                                    // the first index0 should be 0
                                    A_reg_state[i] <= 0;
                                    B_reg_state[i] <= 0;
                                end else begin
                                    // 3,2,1
                                    A_reg_state[i] <= ARRAY_SIZE - i;
                                    B_reg_state[i] <= ARRAY_SIZE - i;
                                end
                            end
                            C_write_cnt <= 0;
                            state <= WORK;
                        end
                    end else if (C_write_cnt <= 4) begin
                        C_write_cnt <= C_write_cnt + 1;
                    end else if (C_write_cnt > 4) begin
                        state <= ERROR;
                    end

                end else begin // m + 1 < Mdiv4
                    if (C_write_cnt == 3) begin
                        PE_rst <= 1;
                        k <= 0;
                        n <= n;
                        m <= m + 1;

                        A_index <= (m + 1) * K_in;
                        B_index <= n * K_in;
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            A_in[i] <= 0;
                            B_in[i] <= 0;
                            A_reg[i] <= 32'd0;
                            B_reg[i] <= 32'd0;
                            C_reg[i] <= {128{1'b0}};
                            if (i == 0) begin
                                // the first index0 should be 0
                                A_reg_state[i] <= 0;
                                B_reg_state[i] <= 0;
                            end else begin
                                // 3,2,1
                                A_reg_state[i] <= ARRAY_SIZE - i;
                                B_reg_state[i] <= ARRAY_SIZE - i;
                            end
                        end

                        C_write_cnt <= 0;

                        state <= WORK;
                    end else if (C_write_cnt < 3) begin
                        C_write_cnt <= C_write_cnt + 1;
                    end else if (C_write_cnt > 3) begin
                        state <= ERROR;
                    end
                end

            end

            DONE: begin
                C_wr_en <= 0;
                busy <= 1'd0;
                state <= IDLE;
            end

            ERROR: begin
              state <= ERROR;
            end

            default: state <= IDLE;
        endcase
    end

end


Systolic_Array#(
    .ARRAY_SIZE(ARRAY_SIZE)
  )  SA(
    .clk             (clk),
    .reset           (reset),
    .PE_rst          (PE_rst),
    // .left            (A_in),
    // .top             (B_in),
    // .out             (C_SA_out)
    .left_0          (A_in[0]),
    .left_1          (A_in[1]),
    .left_2          (A_in[2]),
    .left_3          (A_in[3]),
    .top_0           (B_in[0]),
    .top_1           (B_in[1]),
    .top_2           (B_in[2]),
    .top_3           (B_in[3]),
    .out_0           (C_SA_out[0]),
    .out_1           (C_SA_out[1]),
    .out_2           (C_SA_out[2]),
    .out_3           (C_SA_out[3])
);

endmodule
