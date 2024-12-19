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
    output reg   [11:0] A_idx,
    output       [31:0] A_data_i,
    input signed [31:0] A_data_o,

    output              B_wr_en,
    output reg   [11:0] B_idx,
    output       [31:0] B_data_i,
    input signed [31:0] B_data_o,

    output reg          C_wr_en,
    output reg   [11:0] C_idx,
    output reg  [127:0] C_data_i,
    input       [127:0] C_data_o,
    input signed [31:0] offset,
    output reg    [2:0] C_write_cnt

);
reg PE_rst;
reg   [2:0] state;
localparam IDLE = 3'd0;
localparam PROC = 3'd1;
localparam OUTPUT = 3'd2;
localparam DONE = 3'd3;
localparam WAIT = 3'd4;

assign A_wr_en = 0;
assign B_wr_en = 0;
assign A_data_i = 32'd0;
assign B_data_i = 32'd0;

reg [7:0] m, k, n;
reg [7:0] K_i, M_i, N_i;
reg [7:0] Mdiv4, Ndiv4;
reg signed [8:0] A_i [ARRAY_SIZE-1:0];
reg signed [7:0] B_i [ARRAY_SIZE-1:0];
wire signed [127:0] C_SA_o [ARRAY_SIZE-1:0];


reg signed [31:0] A_Reg [ARRAY_SIZE-1:0];
reg signed [31:0] B_Reg [ARRAY_SIZE-1:0];
reg signed [127:0] C_Reg [ARRAY_SIZE-1:0];

reg [1:0] A_Reg_state [ARRAY_SIZE-1:0];
reg [1:0] B_Reg_state [ARRAY_SIZE-1:0];
reg [2:0] C_Reg_state;

always @(posedge clk) begin
    if (reset) begin
        Mdiv4 <= 0;
        Ndiv4 <= 0;
        K_i <= 0;
        M_i <= 0;
        N_i <= 0;
    end else if (in_valid) begin
        K_i = K;
        M_i = M;
        N_i = N;
        if (M_i <= 4) begin
            Mdiv4 = 1;
        end else if (M_i[1:0] == 2'b00) begin
            Mdiv4 = M_i/4;
        end else begin
            Mdiv4 = M_i/4 + 1;
        end
        if (N_i <= 4) begin
            Ndiv4 = 1;
        end else if (N_i[1:0] == 2'b00) begin
            Ndiv4 = N_i/4;
        end else begin
            Ndiv4 = N_i/4 + 1;
        end
    end

end





integer i;
always @(posedge clk) begin
    if (reset) begin
        PE_rst <= 1;
        state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_idx <= 0;
        B_idx <= 0;
        C_idx <= 0;
        C_write_cnt <= 0;
        busy <= 0;
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            A_i[i] <= 0;
            B_i[i] <= 0;
            A_Reg[i] <= 32'd0;
            B_Reg[i] <= 32'd0;
            C_Reg[i] <= {128{1'b0}};
            A_Reg_state[i] <= 0;
            B_Reg_state[i] <= 0;
        end
        C_Reg_state <= 3'd0;
    end else if (in_valid) begin
        PE_rst <= 1;
        busy <= 1;
        state <= IDLE;
        m <= 0;
        k <= 0;
        n <= 0;
        A_idx <= 0;
        B_idx <= 0;
        C_idx <= 0;
        C_write_cnt <= 0;
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            A_Reg[i] <= 32'd0;
            B_Reg[i] <= 32'd0;
            C_Reg[i] <= {128{1'b0}};
            if (i == 0) begin
                // the first idx0 should be 0
                A_Reg_state[i] <= 0;
                B_Reg_state[i] <= 0;
            end else begin
                A_Reg_state[i] <= ARRAY_SIZE - i;
                B_Reg_state[i] <= ARRAY_SIZE - i;
            end
        end
        C_Reg_state <= 3'd0;
    end else if (busy) begin
        case (state)
            IDLE: begin
                if (busy) begin
                    state <= PROC;
                end else begin
                    state <= IDLE;
                end
            end
            PROC: begin
                PE_rst <= 0;
                C_wr_en <= 0;
                A_Reg[A_Reg_state[0]] = k < K_i? A_data_o : 0;
                B_Reg[B_Reg_state[0]] = k < K_i? B_data_o : 0;
                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                    if (i == 0) begin
                        // specially processing A_i[0]
                        if (k < K_i) begin
                            A_i[0] <= $signed(A_data_o[31:24]) + $signed(offset);
                            B_i[0] <= $signed(B_data_o[31:24]);
                        end else begin
                            A_i[0] <= 0;
                            B_i[0] <= 0;
                        end
                    end else begin
                        if (k < K_i + i) begin
                            A_i[i] <= $signed(A_Reg[A_Reg_state[i]][(8 * (3 - (i % 4))) + 7 -: 8]) + $signed(offset);
                            B_i[i] <= $signed(B_Reg[B_Reg_state[i]][(8 * (3 - (i % 4))) + 7 -: 8]) ;
                        end else begin
                            A_i[i] <= 0;
                            B_i[i] <= 0;
                        end
                    end
                end
                for (i = ARRAY_SIZE-1; i > 0; i = i - 1) begin
                    A_Reg_state[i] <= A_Reg_state[i-1];
                    B_Reg_state[i] <= B_Reg_state[i-1];
                end
                // the last one move to the first one
                A_Reg_state[0] <= A_Reg_state[ARRAY_SIZE-1];
                B_Reg_state[0] <= B_Reg_state[ARRAY_SIZE-1];
                A_idx <= m * K_i + k + 1;
                B_idx <= n * K_i + k + 1;
                k <= k + 1;
                if (k + 3 > K_i + 6) begin
                    C_Reg_state <= 2'd0;
                    C_wr_en <= 1;
                    state <= OUTPUT;
                end
            end
            OUTPUT: begin
                C_idx = n * M_i + m * 4 + C_write_cnt;
                C_data_i = C_SA_o[C_write_cnt];
                if (m + 1 == Mdiv4) begin
                    if (m * 4 + C_write_cnt >= M_i - 1) begin
                        if (n + 1> Ndiv4) begin
                            state <= DONE;
                        end else begin
                            PE_rst <= 1;
                            k <= 0;
                            m <= 0;
                            n <= n + 1;
                            A_idx <= 0;
                            B_idx <= (n + 1) * K_i;
                            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                                A_i[i] <= 0;
                                B_i[i] <= 0;
                                A_Reg[i] <= 32'd0;
                                B_Reg[i] <= 32'd0;
                                C_Reg[i] <= {128{1'b0}};
                                if (i == 0) begin
                                    // the first idx0 should be 0
                                    A_Reg_state[i] <= 0;
                                    B_Reg_state[i] <= 0;
                                end else begin
                                    // 3,2,1
                                    A_Reg_state[i] <= ARRAY_SIZE - i;
                                    B_Reg_state[i] <= ARRAY_SIZE - i;
                                end
                            end
                            C_write_cnt <= 0;
                            state <= PROC;
                        end
                    end else if (C_write_cnt <= 4) begin
                        C_write_cnt <= C_write_cnt + 1;
                    end else if (C_write_cnt > 4) begin
                        state <= WAIT;
                    end
                end else begin
                    if (C_write_cnt == 3) begin
                        PE_rst <= 1;
                        k <= 0;
                        n <= n;
                        m <= m + 1;
                        A_idx <= (m + 1) * K_i;
                        B_idx <= n * K_i;
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            A_i[i] <= 0;
                            B_i[i] <= 0;
                            A_Reg[i] <= 32'd0;
                            B_Reg[i] <= 32'd0;
                            C_Reg[i] <= {128{1'b0}};
                            if (i == 0) begin
                                // the first idx0 should be 0
                                A_Reg_state[i] <= 0;
                                B_Reg_state[i] <= 0;
                            end else begin
                                // 3,2,1
                                A_Reg_state[i] <= ARRAY_SIZE - i;
                                B_Reg_state[i] <= ARRAY_SIZE - i;
                            end
                        end
                        C_write_cnt <= 0;
                        state <= PROC;
                    end else if (C_write_cnt < 3) begin
                        C_write_cnt <= C_write_cnt + 1;
                    end else if (C_write_cnt > 3) begin
                        state <= WAIT;
                    end
                end
            end
            DONE: begin
                C_wr_en <= 0;
                busy <= 1'd0;
                state <= IDLE;
            end
            WAIT: begin
              state <= WAIT;
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
    // .left            (A_i),
    // .top             (B_i),
    // .out             (C_SA_o)
    .left_0          (A_i[0]),
    .left_1          (A_i[1]),
    .left_2          (A_i[2]),
    .left_3          (A_i[3]),
    .top_0           (B_i[0]),
    .top_1           (B_i[1]),
    .top_2           (B_i[2]),
    .top_3           (B_i[3]),
    .out_0           (C_SA_o[0]),
    .out_1           (C_SA_o[1]),
    .out_2           (C_SA_o[2]),
    .out_3           (C_SA_o[3])
);

endmodule
