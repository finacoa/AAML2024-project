`include "PE.v"

module Systolic_Array #(parameter ARRAY_SIZE = 4)(
    input              clk,
    input              reset,
    input              PE_rst,
    // input signed [8:0] left[ARRAY_SIZE-1:0],
    // input signed [7:0] top[ARRAY_SIZE-1:0],
    // output reg [127:0] out[ARRAY_SIZE-1:0]
    input signed [8:0] left_0,
    input signed [8:0] left_1,
    input signed [8:0] left_2,
    input signed [8:0] left_3,
    input signed [7:0] top_0,
    input signed [7:0] top_1,
    input signed [7:0] top_2,
    input signed [7:0] top_3,
    output reg [127:0] out_0,
    output reg [127:0] out_1,
    output reg [127:0] out_2,
    output reg [127:0] out_3
    );

// parameter ARRAY_SIZE = 4;
wire [8:0] right_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [7:0] bottom_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [31:0] acc_reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];


// output to C
// always @(posedge clk) begin
//     integer i;
//     for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
//         out[i] = {acc_reg[i][0], acc_reg[i][1], acc_reg[i][2], acc_reg[i][3]};
//     end
// end


always @(posedge clk) begin
    out_0 = {acc_reg[0][0], acc_reg[0][1], acc_reg[0][2], acc_reg[0][3]};
    out_1 = {acc_reg[1][0], acc_reg[1][1], acc_reg[1][2], acc_reg[1][3]};
    out_2 = {acc_reg[2][0], acc_reg[2][1], acc_reg[2][2], acc_reg[2][3]};
    out_3 = {acc_reg[3][0], acc_reg[3][1], acc_reg[3][2], acc_reg[3][3]};

end

// genvar i, j;
// generate
//     for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
//         for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
//             PE pe (
//                 .clk(clk),
//                 .reset(reset),
//                 .PE_rst(PE_rst),
//                 .left_in((j == 0) ? left[i] : right_reg[i][j-1]),
//                 .top_in((i == 0) ? top[j] : bottom_reg[i-1][j]),
//                 .right_out(right_reg[i][j]),
//                 .bottom_out(bottom_reg[i][j]),
//                 .acc(acc_reg[i][j])
//             );
//         end
//     end
// endgenerate

genvar i, j;
generate
    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
        for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
            PE pe (
                .clk(clk),
                .reset(reset),
                .PE_rst(PE_rst),
                // .offset(offset),
                .left_in((j == 0) ?
                         (i == 0 ? left_0
                        :(i == 1 ? left_1
                        :(i == 2 ? left_2
                        :left_3))
                        ): right_reg[i][j-1]),
                .top_in((i == 0) ?
                        (j == 0 ? top_0
                        : (j == 1 ? top_1
                        : (j == 2 ? top_2
                        : top_3))
                        ) : bottom_reg[i-1][j]),
                .right_out(right_reg[i][j]),
                .bottom_out(bottom_reg[i][j]),
                .acc(acc_reg[i][j])
            );
        end
    end
endgenerate


endmodule
