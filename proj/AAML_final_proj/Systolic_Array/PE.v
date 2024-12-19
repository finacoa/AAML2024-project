module PE(
    input             clk,
    input             reset,
    input             PE_rst,
    input       [8:0] left_i,
    input       [7:0] top_i,
    output reg  [8:0] right_o,
    output reg  [7:0] bottom_o,
    output reg [31:0] acc
    );

    always @(posedge clk) begin
        if (reset) begin
            right_o <= 0;
            bottom_o <= 0;
            acc <= 0;
        end else if (PE_rst) begin
            acc <= 0;
        end else begin
            acc <= ($signed(left_i) * $signed(top_i)) + $signed(acc);
            right_o <= left_i;
            bottom_o <= top_i;
        end
    end
endmodule
