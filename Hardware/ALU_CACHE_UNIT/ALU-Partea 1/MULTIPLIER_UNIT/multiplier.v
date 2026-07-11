module RCA(
  input[7:0] A,
  input[7:0] B,
  input cin,
  output[7:0] sum,
  output cout,
  output overflw
);
wire [7:0] carry;
//
FAC FA0 (.A(A[0]), .B(B[0]), .cin(cin), .sum(sum[0]), .cout(carry[0]));
FAC FA1 (.A(A[1]), .B(B[1]), .cin(carry[0]), .sum(sum[1]), .cout(carry[1]));
FAC FA2 (.A(A[2]), .B(B[2]), .cin(carry[1]), .sum(sum[2]), .cout(carry[2]));
FAC FA3 (.A(A[3]), .B(B[3]), .cin(carry[2]), .sum(sum[3]), .cout(carry[3]));
FAC FA4 (.A(A[4]), .B(B[4]), .cin(carry[3]), .sum(sum[4]), .cout(carry[4]));
FAC FA5 (.A(A[5]), .B(B[5]), .cin(carry[4]), .sum(sum[5]), .cout(carry[5]));
FAC FA6 (.A(A[6]), .B(B[6]), .cin(carry[5]), .sum(sum[6]), .cout(carry[6]));
FAC FA7 (.A(A[7]), .B(B[7]), .cin(carry[6]), .sum(sum[7]), .cout(cout));
assign overflw=cout | carry[6];
endmodule

//==========================================================================

module FAC(
  input A,
  input B,
  input cin,
  output sum,
  output cout
);
assign sum = A ^ B ^ cin;
assign cout = (A & B) | (A & cin) | (B & cin);
endmodule

//==========================================================================

module RCA_Star(
  input[7:0] A,
  input[7:0] B,
  input cin,
  output[7:0] sum,
  output cout,
  output pi
);
wire[7:0] propagate;
wire[7:0] carry;

FAC_Star FAS0 (.A(A[0]), .B(B[0]), .cin(cin), .sum(sum[0]), .cout(carry[0]), .pi(propagate[0]));
FAC_Star FAS1 (.A(A[1]), .B(B[1]), .cin(carry[0]), .sum(sum[1]), .cout(carry[1]), .pi(propagate[1]));
FAC_Star FAS2 (.A(A[2]), .B(B[2]), .cin(carry[1]), .sum(sum[2]), .cout(carry[2]), .pi(propagate[2]));
FAC_Star FAS3 (.A(A[3]), .B(B[3]), .cin(carry[2]), .sum(sum[3]), .cout(carry[3]), .pi(propagate[3]));
FAC_Star FAS4 (.A(A[4]), .B(B[4]), .cin(carry[3]), .sum(sum[4]), .cout(carry[4]), .pi(propagate[4]));
FAC_Star FAS5 (.A(A[5]), .B(B[5]), .cin(carry[4]), .sum(sum[5]), .cout(carry[5]), .pi(propagate[5]));
FAC_Star FAS6 (.A(A[6]), .B(B[6]), .cin(carry[5]), .sum(sum[6]), .cout(carry[6]), .pi(propagate[6]));
FAC_Star FAS7 (.A(A[7]), .B(B[7]), .cin(carry[6]), .sum(sum[7]), .cout(cout), .pi(propagate[7]));

assign pi = propagate[0] & propagate[1] & propagate[2] & propagate[3] & propagate[4] & propagate[5] & propagate[6] & propagate[7];
endmodule

//==========================================================================

module FAC_Star(
  input A,
  input B,
  input cin,
  output sum,
  output cout,
  output pi
  );
  
  assign sum = A ^ B ^ cin;
  assign cout = (A & B) | (A & cin) | (B & cin);
  assign pi =A | B;
endmodule
module makeXor #(
    parameter WIDTH = 32 
)(
    input [WIDTH-1:0] a,
    input b,
    output [WIDTH-1:0] aXor
);

// Internal wire for the result of the XOR operation
wire [WIDTH-1:0] temp_aXor;
//assign aXor={WIDTH{1'd0}};

// Perform XOR operation between each bit of 'a' and 'b'
genvar i;
generate
    for (i = 0; i < WIDTH; i = i + 1) begin : XOR_loop
        assign temp_aXor[i] = a[i] ^ b;
    end
endgenerate

// Assign the result to the output
assign aXor = temp_aXor;
endmodule

//==========================================================================

module CSkA(
    //va trebui 1.x[32],y[32]-->sum[67](adunare)  2.x[33],y[33]-->sum[33](impartire)  3.x[34],y[34]-->sum[34](inmultire)
    input [31:0] 	X,
    input [31:0] 	Y,
    input [32:0] 	X2,
    input [32:0] 	Y2,
    input [33:0] 	X3,
    input [33:0] 	Y3, 
    input 	cin,
    output 	cout,
    output 	cout2,
    output 	cout3,
    output [66:0] sum,
    output [32:0] sum2,
    output [33:0] sum3
);
// Declaration of wires
wire [31:0] Y_xor_cin;
wire [32:0] Y2_xor_cin;
wire [33:0] Y3_xor_cin;
wire ovrflw;

makeXor #(.WIDTH(32)) xor1 (.a(Y), .b(cin), .aXor(Y_xor_cin));
makeXor #(.WIDTH(33)) xor2 (.a(Y2), .b(cin), .aXor(Y2_xor_cin));
makeXor #(.WIDTH(34)) xor3 (.a(Y3), .b(cin), .aXor(Y3_xor_cin));


//pt 32b
wire [2:0] carry;
wire Propagate1,Propagate2;

RCA RCA0(.A(X[7:0]), .B(Y_xor_cin[7:0]), .cin(cin), .sum(sum[7:0]), .cout(carry[0]), .overflw(overflw));
RCA_Star RCA1(.A(X[15:8]), .B(Y_xor_cin[15:8]), .cin(carry[0]), .sum(sum[15:8]), .cout(carry[1]), .pi(Propagate1));
RCA_Star RCA2(.A(X[23:16]), .B(Y_xor_cin[23:16]), .cin(carry[1]|(Propagate1&carry[0])), .sum(sum[23:16]), .cout(carry[2]), .pi(Propagate2));
RCA RCA3(.A(X[31:24]), .B(Y_xor_cin[31:24]), .cin(carry[2]|(Propagate2&carry[1])), .sum(sum[31:24]), .cout(cout), .overflw(overflw));

genvar i;
generate
    for (i = 32; i <67; i = i + 1) begin :loop
        assign sum[i]=sum[31];
    end
endgenerate

//pt 33b
wire [2:0] carry2;
wire Propagate21,Propagate22;
wire auxcout;

RCA RCA10(.A(X2[7:0]), .B(Y2_xor_cin[7:0]), .cin(cin), .sum(sum2[7:0]), .cout(carry2[0]), .overflw(overflw));
RCA_Star RCA11(.A(X2[15:8]), .B(Y2_xor_cin[15:8]), .cin(carry2[0]), .sum(sum2[15:8]), .cout(carry2[1]), .pi(Propagate21));
RCA_Star RCA12(.A(X2[23:16]), .B(Y2_xor_cin[23:16]), .cin(carry2[1]|(Propagate21 & carry2[0])), .sum(sum2[23:16]), .cout(carry2[2]), .pi(Propagate22));
RCA RCA13(.A(X2[31:24]), .B(Y2_xor_cin[31:24]), .cin(carry2[2]|(Propagate22 & carry2[1])), .sum(sum2[31:24]), .cout(auxcout), .overflw(overflw));
FAC FAC11(.A(X2[32]), .B(Y2_xor_cin[32]), .cin(auxcout), .sum(sum2[32]), .cout(cout2)); // Connect carry-out of RCA13 to cin of FAC11

//pt 34b
wire [2:0] carry3;
wire Propagate31,Propagate32;
wire auxcout1,auxcout2;

RCA RCA20(.A(X3[7:0]), .B(Y3_xor_cin[7:0]), .cin(cin), .sum(sum3[7:0]), .cout(carry3[0]), .overflw(overflw));
RCA_Star RCA21(.A(X3[15:8]), .B(Y3_xor_cin[15:8]), .cin(carry3[0]), .sum(sum3[15:8]), .cout(carry3[1]), .pi(Propagate31));
RCA_Star RCA22(.A(X3[23:16]), .B(Y3_xor_cin[23:16]), .cin(carry3[1]|(Propagate31 & carry3[0])), .sum(sum3[23:16]), .cout(carry3[2]), .pi(Propagate32));
RCA RCA23(.A(X3[31:24]), .B(Y3_xor_cin[31:24]), .cin(carry3[2]|(Propagate32 & carry3[1])), .sum(sum3[31:24]), .cout(auxcout1), .overflw(overflw));
FAC FAC21(.A(X3[32]), .B(Y3_xor_cin[32]), .cin(auxcout1), .sum(sum3[32]), .cout(auxcout2)); // Connect carry-out of RCA23 to cin of FAC21
FAC FAC22(.A(X3[33]), .B(Y3_xor_cin[33]), .cin(auxcout2), .sum(sum3[33]), .cout(cout3)); // Connect carry-out of FAC21 to cin of FAC2   
endmodule

//==========================================================================

module controlUnit(
input clk,rst_b,
input START,
input [2:0] cnt,
input w,x,y,z,
output reg [7:0] cSig
);
localparam s0 = 1;
localparam s1 = 2;
localparam s2 = 4;
localparam s3 = 8;
localparam s4 = 16;
localparam s5 = 32;
localparam s6 = 64;
localparam s7 = 128;
localparam s8 = 256;
localparam s9 = 512;
localparam s10 = 1024;
localparam s11 = 2048;
localparam s12 = 4096;
localparam s13 = 8192;

reg [14:0] cst,nxst;

always @* begin
case(cst)
s0:begin
if(START==1'b1)
nxst=s1;
else
nxst=s0;
end

s1:begin
nxst=s2;
end

s2:begin
nxst=s3;
end

s3:begin
if( ((w)&(x)&(y)&(z)) | ((~w)&(~x)&(~y)&(~z)) )//0
nxst=s12;
else
if( ((~w)&(~x)&(~y)&(z)) | ((~w)&(~x)&(y)&(~z)) )//1
nxst=s4;
else
if( ((w)&(x)&(~y)&(z)) | ((w)&(x)&(y)&(~z)) )//-1
nxst=s5;
else
if( ((~w)&(~x)&(y)&(z)) | ((~w)&(x)&(~y)&(~z)) )//2
nxst=s6;
else
if( ((w)&(x)&(~y)&(~z)) | ((w)&(~x)&(y)&(z)) )//-2
nxst=s7;
else
if( ((~w)&(x)&(~y)&(z)) | ((~w)&(x)&(y)&(~z)) )//3
nxst=s8;
else
if( ((w)&(~x)&(~y)&(z)) | ((w)&(~x)&(y)&(~z)) )//-3
nxst=s9;
else
if((~w)&(x)&(y)&(z))//4
nxst=s10;
else
if((w)&(~x)&(~y)&(~z))//-4
nxst=s11;
end

s4:begin
nxst=s12;
end
s5:begin
nxst=s12;
end
s6:begin
nxst=s12;
end
s7:begin
nxst=s12;
end
s8:begin
nxst=s12;
end
s9:begin
nxst=s12;
end
s10:begin
nxst=s12;
end
s11:begin
nxst=s12;
end

s12:begin
if(cnt[0]&cnt[1]&cnt[2])
nxst=s13;
else
nxst=s3;
end

s13:begin
nxst=s0;
end
endcase
end

always @* begin
    case (cst)
        s1:cSig=8'd1;
        s2:cSig=8'd2;
        
        s3:begin
        if( ((w)&(x)&(y)&(z)) | ((~w)&(~x)&(~y)&(~z)) )
        cSig=8'd32;  
        end
        
        s4:cSig=8'd160;
        s5:cSig=8'd164;
        s6:cSig=8'd176;
        s7:cSig=8'd180;
        s8:cSig=8'd168;
        s9:cSig=8'd172;
        s10:cSig=8'd184;
        s11:cSig=8'd188;
        s13:cSig=8'd64;
        
        default: cSig=8'd0;
    endcase
end

always @(posedge clk,negedge rst_b) begin
//$display("q = %b", cst);
if(!rst_b)begin
cst<=s0;
end
else
cst<=nxst;       
end
endmodule

//===========================================================================================

module mux4to1(
    input [33:0] data_in0,
    input [33:0] data_in1,
    input [33:0] data_in2,
    input [33:0] data_in3,
    input [1:0] select,
    output reg [33:0] data_out
);
always @* begin
    case (select)
        2'b00: data_out = data_in0;
        2'b01: data_out = data_in1;
        2'b10: data_out = data_in2;
        2'b11: data_out = data_in3;
        default: data_out = 34'bx; // Handle invalid select values
    endcase
end
endmodule

module mux2to1A(
    input [33:0] data_in0,
    input [33:0] data_in1,
    input select,
    output reg [33:0] data_out
);
always @* begin
    case (select)
        1'b0: data_out = data_in0;
        1'b1: data_out = data_in1;
        default: data_out = 34'bX; 
    endcase
end
endmodule

module mux2to1B(
    input [32:0] data_in0,
    input [32:0] data_in1,
    input select,
    output reg [32:0] data_out
);
always @* begin
    case (select)
        1'b0: data_out = data_in0;
        1'b1: data_out = data_in1;
        default: data_out = 33'bX; 
    endcase
end
endmodule

//===========================================================================================

module operations(
    input c7,
    input clk,
    input [33:0] m,
    input [33:0] a,
    input [2:0] cSig,
    output reg [33:0] newa
);
    reg [33:0] twoM,fourM;
    wire [33:0] threeM,aux,aux2,sum3;
    wire cout, cout2, cout3;
    wire [66:0] sum;
    wire [32:0] sum2;

always @* begin 
twoM=m<<1;
fourM=m<<2;
end
CSkA CSkA_inst2 (
    .X(32'b0),          // Set X to zero, unused
    .Y(32'b0),          // Set Y to zero, unused
    .X2(33'b0),            // Connect X2
    .Y2(33'b0),            // Connect Y2
    .X3(twoM),         // Set X3 to zero, unused
    .Y3(m),         // Set Y3 to zero, unused
    .cin(1'b0),
    .cout(cout),
    .cout2(cout2),
    .cout3(cout3),
    .sum(sum),
    .sum2(sum2),
    .sum3(threeM)
);
mux4to1 inst1(.data_in0(m), .data_in1(twoM), .data_in2(threeM), .data_in3(fourM), .select({cSig[1],cSig[2]}), .data_out(aux));
         CSkA CSkA_inst1 (
                    .X(32'b0),
                    .Y(32'b0),
                    .X2(33'b0),
                    .Y2(33'b0),
                    .X3(a),
                    .Y3(aux),
                    .cin(cSig[0]),
                    .cout(cout),
                    .cout2(cout2),
                    .cout3(cout3),
                    .sum(sum),
                    .sum2(sum2),
                    .sum3(aux2)
                );
mux2to1A selectFinal(.data_in0(a), .data_in1(aux2), .select(c7), .data_out(sum3));
always @* begin
//if(c7)begin
//$display("op=%b\nthreeM=%b",aux,threeM);end
    newa <= sum3;
end   
endmodule

//===========================================================================================

module lshift(
input c5,
input clk,
input [33:0] a,
input [32:0] q,
input qNeg,
output reg [33:0] aOUT,
output reg [32:0] qOUT,
output reg qNegOUT
);
wire [33:0] aOUTR;
wire [33:0] aux1;
wire [32:0] qOUTR;
wire [32:0] aux2;

assign aOUTR=a>>3;
assign qOUTR=q>>3;

mux2to1A inst1(.data_in0(a), .data_in1(aOUTR), .select(c5), .data_out(aux1));
mux2to1B inst2(.data_in0(q), .data_in1(qOUTR), .select(c5), .data_out(aux2));

always @* begin
aOUT=aux1;qOUT=aux2;qNegOUT=qNeg;
if(c5)begin
qNegOUT=q[2];
aOUT[33:31]={a[33], a[33], a[33]};
qOUT[32:30]={a[2], a[1], a[0]}; 
end
end
endmodule

//===========================================================================================

module counter (
    input clk,      // Clock input
    input c_up,     // Count up enable
    input rst,      // Reset input (active low)
    input clr,
    input[2:0] count_reg,
    output reg [2:0] count  // 8-bit counter output
);
// Define counter behavior
always @(posedge c_up,posedge clr,negedge rst)begin
    if (!rst) begin
        // Reset the counter to 0 when rst is asserted (active low)
        count <= 3'd0;
    end else if (clr) begin
        // Clear the counter to 0 when clr is asserted
        count <= 3'd0;
    end else if (c_up) begin
        // Increment the counter if count up is enabled
        count <= count_reg + 1;
    end
end
endmodule

//=================================================================================

module fshift(
input clk,
input active,
input [33:0] a,
input [32:0] q, 
output reg [33:0] newa,
output reg [32:0] newq
);
wire [33:0] aOUTR;
wire [33:0] aux1;
wire [32:0] qOUTR;
wire [32:0] aux2;

assign aOUTR=a>>12;
assign qOUTR=q>>12;

mux2to1A inst1(.data_in0(a), .data_in1(aOUTR), .select(active), .data_out(aux1));
mux2to1B inst2(.data_in0(q), .data_in1(qOUTR), .select(active), .data_out(aux2));

always @* begin
newa=aux1;newq=aux2;
if(active)begin
newa[33:22]={a[33], a[33], a[33],
    a[33], a[33], a[33],
    a[33], a[33], a[33],a[33],a[33],a[33]
};
newq[32:21]={a[11],a[10], a[9], a[8],
a[7], a[6], a[5],
    a[4], a[3], a[2], a[1],a[0]
}; 
end
end
endmodule

//==========================================================================

module multiplier(
input [31:0] X,Y,
input clk,
input active,//formula lui OP
output reg [66:0] product,
output reg suff
);
reg [33:0] a;
reg [32:0] q;reg qNegREG;
reg [33:0] m;
reg [2:0] counter;
reg activeREG;

wire [33:0] aAux,aAux2,aAux3;
wire [32:0] qAux,qAux2;wire qNeg;
wire [2:0] counterAux;
wire [7:0] cSig;

reg rst=0,sec=0;

always @(posedge clk) begin
    if(!rst) begin
    suff<=0;
    a <= 34'd0;
    counter <= 3'd0;
    q <= {X[31], X};qNegREG<=0;
    m <= {{2{Y[31]}}, Y};
     activeREG <= active; 
    end
end

controlUnit reff1(.clk(clk), .rst_b(rst), .START(activeREG), .cnt(counter), .w(q[2]), .x(q[1]), .y(q[0]), .z(qNegREG), .cSig(cSig));

operations op(.m(m), .clk(clk), .a(a), .cSig({cSig[4],cSig[3],cSig[2]}), .newa(aAux), .c7(cSig[7]));
lshift inst(.a(aAux), .clk(clk), .q(q), .qNeg(qNegREG), .aOUT(aAux2), .qOUT(qAux), .qNegOUT(qNeg), .c5(cSig[5]));
counter inst0(.clk(clk), .c_up(cSig[5]), .rst(rst), .clr(cSig[0]), .count_reg(counter), .count(counterAux));
fshift finale(.clk(clk),.active(cSig[6])
,.a(aAux2),.q(qAux),.newa(aAux3),.newq(qAux2)
);
always @(posedge clk) begin
  if(!activeREG)suff=1;
    a <= aAux3;
    q <= qAux2;qNegREG<=qNeg;
    counter <= counterAux;
    if(cSig[6])activeREG=0;
    product <= {a,q};
    rst=sec;
    sec=1;
    //$display("x=%b\ny=%b\n\nrst=%b\nactiveREG=%b\ncSig=%b\ncounterAux=%b\na=%b\nq=%b qNeg=%b\n\nsuff=%b\n",X,Y,rst,activeREG,cSig,counterAux,aAux2,qAux,qNeg,suff);
end
endmodule











module multiplier_tb;
  
  reg [31:0] X,Y;
  reg clk,enable;
  wire [66:0] product;
  wire suff;
  
  multiplier multiplier_inst (.suff(suff),
   .X(X),.Y(Y),.clk(clk),.active(enable),.product(product)
  );

  // Stimulus
  initial begin
    // Initialize inputs
    //$monitor("product = %b\nsuff=%b\n\n", product,suff);
    //x 0 0 0 0100 1000
    //y 0 0 0 0101 1001
    X = 32'd172; // Example input value
    Y = 32'd172; // Example input value
    enable=1'b1;
    // Wait some time
    #1310;

    // End simulation
    $finish;
  end
  
localparam run_cycle=10,cycles=65;
initial begin
  clk=1'b0;
  repeat (cycles*2)
#run_cycle clk=~clk;
end
  
endmodule
