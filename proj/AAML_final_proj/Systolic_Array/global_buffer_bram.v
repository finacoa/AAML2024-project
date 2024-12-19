module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input      [ADDR_BITS-1:0] idx,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (negedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[idx] <= data_in;
      end else begin
        data_out <= gbuff[idx];
      end
    end
  end

endmodule
