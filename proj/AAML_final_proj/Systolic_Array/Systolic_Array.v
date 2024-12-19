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
wire [8:0] right [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [7:0] bottom [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
wire [31:0] acc [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];


// output to C
// always @(posedge clk) begin
//     integer i;
//     for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
//         out[i] = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
//     end
// end


always @(posedge clk) begin
    out_0 = {acc[0][0], acc[0][1], acc[0][2], acc[0][3]};
    out_1 = {acc[1][0], acc[1][1], acc[1][2], acc[1][3]};
    out_2 = {acc[2][0], acc[2][1], acc[2][2], acc[2][3]};
    out_3 = {acc[3][0], acc[3][1], acc[3][2], acc[3][3]};

end

// genvar i, j;
// generate
//     for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
//         for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
//             PE pe (
//                 .clk(clk),
//                 .reset(reset),
//                 .PE_rst(PE_rst),
//                 .left_i((j == 0) ? left[i] : right[i][j-1]),
//                 .top_i((i == 0) ? top[j] : bottom[i-1][j]),
//                 .right_o(right[i][j]),
//                 .bottom_o(bottom[i][j]),
//                 .acc(acc[i][j])
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
                .left_i((j == 0) ?
                        (i == 0 ? left_0
                        :(i == 1 ? left_1
                        :(i == 2 ? left_2
                        :left_3))
                        ): right[i][j-1]),
                .top_i((i == 0) ?
                        (j == 0 ? top_0
                        : (j == 1 ? top_1
                        : (j == 2 ? top_2
                        : top_3))
                        ) : bottom[i-1][j]),
                .right_o(right[i][j]),
                .bottom_o(bottom[i][j]),
                .acc(acc[i][j])
            );
        end
    end
endgenerate


endmodule
