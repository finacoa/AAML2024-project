module PE(
    input             clk,
    input             reset,
    input             PE_rst,
    input       [8:0] left_in,
    input       [7:0] top_in,
    output reg  [8:0] right_out,
    output reg  [7:0] bottom_out,
    output reg [31:0] acc
    );

    always @(posedge clk) begin
        if (reset) begin
            right_out <= 9'd0;
            bottom_out <= 8'd0;
            acc <= 32'd0;
        end else if (PE_rst) begin
            acc <= 32'd0;
        end else begin
            acc <= ($signed(left_in) * $signed(top_in)) + $signed(acc);
            right_out <= left_in;
            bottom_out <= top_in;
        end
    end
endmodule
