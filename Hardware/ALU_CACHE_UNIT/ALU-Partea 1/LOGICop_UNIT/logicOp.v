module and_op(
    input [31:0] a,
    input [31:0] b,
    output [66:0] c
);
genvar i;
generate 
    for(i = 0; i < 32; i = i + 1) begin
        assign c[i] = a[i] & b[i];
    end
    for(i = 32; i < 67; i = i + 1) begin
        assign c[i] = 0;
    end
endgenerate
endmodule

//====================================================

module xor_op(
    input [31:0] a,
    input [31:0] b,
    output [66:0] c
);
genvar i;
generate 
    for(i = 0; i < 32; i = i + 1) begin
        assign c[i] = a[i] ^ b[i];
    end
    for(i = 32; i < 67; i = i + 1) begin
        assign c[i] = 0;
    end
endgenerate
endmodule

//=====================================================

module or_op(
    input [31:0] a,
    input [31:0] b,
    output [66:0] c
);
genvar i;
generate 
    for(i = 0; i < 32; i = i + 1) begin
        assign c[i] = a[i] | b[i];
    end
    for(i = 32; i < 67; i = i + 1) begin
        assign c[i] = 0;
    end
endgenerate
endmodule

//====================================================

module logicOp(
input [31:0] X,Y,
output [66:0] shiftedRX,shiftedLX,
output [66:0] shiftedRY,shiftedLY,
output [66:0] andOp,orOp,xorOp,
output suff
);
wire [66:0] X2,Y2;
assign X2={{35{X[31]}},X};
assign Y2={{35{Y[31]}},Y};

assign shiftedLX=X2<<32;
assign shiftedRX=X2>>32;

assign shiftedLY=Y2<<32;
assign shiftedRY=Y2>>32;

and_op inst1(.a(X), .b(Y), .c(andOp));
or_op inst2(.a(X), .b(Y), .c(orOp));
xor_op inst3(.a(X), .b(Y), .c(xorOp));
assign suff=1;
endmodule;











/*
module logicOp_tb;
reg [31:0] X,Y;reg clk;
wire [66:0] shiftedRX,shiftedLX;
wire [66:0] shiftedRY,shiftedLY;
wire [66:0] andOp,orOp,xorOp;
wire suff;

  logicOp logic_inst(
     .X(X),.Y(Y), 
     .shiftedRY(shiftedRY), .shiftedLY(shiftedLY), 
     .shiftedRX(shiftedRX), .shiftedLX(shiftedLX), 
     .andOp(andOp), .orOp(orOp), .xorOp(xorOp), .suff(suff)
  );

initial begin
    // Seteaz? valorile pentru x1, y1, x2, y2, x3 ?i y3
    X = 32'b01000000000000010000000000000000;
    Y = 32'b01010000000001010000000000000000;
    // Afi?eaz? valorile rezultate
    $monitor("RShiftX:%b\nLShiftX=%b\n\nRShiftY:%b\nLShiftY=%b\n\nand=%b\n or=%b\nxor=%b\nsuff=%b\n\n", shiftedRX,shiftedLX,shiftedRY,shiftedLY,andOp,orOp,xorOp,suff);
end

localparam run_cycle=10,cycles=65;
initial begin
  clk=1'b0;
  repeat (cycles*2)
#run_cycle clk=~clk;
end
endmodule
*/
