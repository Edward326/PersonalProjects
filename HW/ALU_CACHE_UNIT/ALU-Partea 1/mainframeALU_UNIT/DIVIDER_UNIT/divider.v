module RCA(
  input[7:0] A,
  input[7:0] B,
  input cin,
  output[7:0] sum,
  output cout,
  output overflw
);

wire [7:0] carry;

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

//==========================================================================

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
  output [33:0] sum3,
  output suff
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

assign suff=1;
endmodule




















module controlUnit(
input clk,rst_b,
input START,
input [4:0] cnt,
input w,
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

reg [10:0] cst,nxst;

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
nxst=s4;
end

s4:begin
nxst=s5;
end

s5:begin
if(~w)
nxst=s7;
else
nxst=s6;
end

s6:begin
nxst=s8;
end
s7:begin
nxst=s8;
end

s8:begin
if(cnt[0]&cnt[1]&cnt[2]&cnt[3]&cnt[4])
nxst=s10;
else
nxst=s9;
end

s9:begin
nxst=s4;
end

s10:begin
nxst=s0;
end
endcase
end

always @* begin
    case (cst)
        s1:cSig=8'd1;
        s2:cSig=8'd2;
        s3:cSig=8'd4;  
        s4:cSig=8'd24;
        s6:cSig=8'd8;
        s7:cSig=8'd32;
        s9:cSig=8'd64;
        s10:cSig=8'd128;
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

module mux2to1A(
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

module mux2to1B(
    input [31:0] data_in0,
    input [31:0] data_in1,
    input select,
    output reg [31:0] data_out
);
always @* begin
    case (select)
        1'b0: data_out = data_in0;
        1'b1: data_out = data_in1;
        default: data_out = 32'bX; 
    endcase
end
endmodule

//===========================================================================================

module operations(
    input clk,
    input [32:0] m,
    input [32:0] a,
    input [1:0] cSig,
    output reg [32:0] newa
);
    wire cout, cout2, cout3;
    wire [66:0] sum;
    wire [32:0] sum2,aux;
    wire [33:0] sum3;
    wire suff;

CSkA CSkA_inst2 (.suff(suff),
    .X(32'd0),          // Set X to zero, unused
    .Y(32'd0),          // Set Y to zero, unused
    .X2(a),            // Connect X2
    .Y2(m),            // Connect Y2
    .X3(34'd0),         // Set X3 to zero, unused
    .Y3(34'd0),         // Set Y3 to zero, unused
    .cin(cSig[1]),
    .cout(cout),
    .cout2(cout2),
    .cout3(cout3),
    .sum(sum),
    .sum2(sum2),
    .sum3(sum3)
);
mux2to1A selectFinal(.data_in0(a), .data_in1(sum2), .select(cSig[0]), .data_out(aux));

always @* begin
//if(cSig[0])begin
//$display("sum=%b\n",sum2);end
    newa <= aux;
end   
endmodule

//===========================================================================================

module checkNotRestore(
input clk,
input c5,
input [31:0] q,
output reg [31:0] newq
);
always @* begin
newq<=q;
if(c5)
newq[0]<=c5;
end
endmodule

//===========================================================================================

module rshift(
input c6,
input clk,
input [32:0] a,
input [31:0] q,
output reg [32:0] aOUT,
output reg [31:0] qOUT
);
wire [32:0] aOUTR;
wire [32:0] aux1;
wire [31:0] qOUTR;
wire [31:0] aux2;

assign aOUTR=a<<1;
assign qOUTR=q<<1;

mux2to1A inst1(.data_in0(a), .data_in1(aOUTR), .select(c6), .data_out(aux1));
mux2to1B inst2(.data_in0(q), .data_in1(qOUTR), .select(c6), .data_out(aux2));

always @* begin
aOUT=aux1;qOUT=aux2;
if(c6)begin
aOUT[0]=q[31];
qOUT[0]=1'b0; 
end
end
endmodule

//===========================================================================================

module counter (
    input clk,      // Clock input
    input c_up,     // Count up enable
    input rst,      // Reset input (active low)
    input clr,
    input[4:0] count_reg,
    output reg [4:0] count  // 8-bit counter output
);
// Define counter behavior
always @(posedge c_up,posedge clr,negedge rst)begin
    if (!rst) begin
        // Reset the counter to 0 when rst is asserted (active low)
        count <= 5'd0;
    end else if (clr) begin
        // Clear the counter to 0 when clr is asserted
        count <= 5'd0;
    end else if (c_up) begin
        // Increment the counter if count up is enabled
        count <= count_reg + 1;
    end
end
endmodule

module divider(
input [31:0] X,Y,
input clk,
input active,//formula lui OP
output reg [66:0] quatient,//Q
output reg [32:0] remainder,//A
output reg suff
);
reg [32:0] a;
reg [31:0] q;
reg [32:0] m;
reg [4:0] counter;
reg activeREG;

wire [32:0] aAux,aAux2,aAux3;
wire [31:0] qAux,qAux2,qAux3;
wire [4:0] counterAux;
wire [7:0] cSig;

reg rst=0,sec=0;

always @(posedge clk) begin
    if(!rst) begin
    suff<=0;
    a <= 33'd0;
    counter <= 5'd0;
    q <= X;
    m <= {Y[31], Y};
     activeREG <= active; 
    end
end

controlUnit fsm(.clk(clk), .rst_b(rst), .START(activeREG), .cnt(counter), .w(a[32]), .cSig(cSig));
rshift fshift(.a(a), .clk(clk), .q(q), .aOUT(aAux), .qOUT(qAux), .c6(cSig[2]));
operations op(.m(m), .clk(clk), .a(aAux), .cSig({cSig[4],cSig[3]}), .newa(aAux2));
checkNotRestore check(.clk(clk), .q(qAux), .c5(cSig[5]), .newq(qAux2));
rshift sshisft(.a(aAux2), .clk(clk), .q(qAux2), .aOUT(aAux3), .qOUT(qAux3), .c6(cSig[6]));
counter cntUp(.clk(clk), .c_up(cSig[6]), .rst(rst), .clr(cSig[0]), .count_reg(counter), .count(counterAux));

always @(posedge clk) begin
//$display("x=%b\ny=%b\n\nrst=%b\nactiveREG=%b\ncSig=%b\ncounterAux=%b\na=%b\nq=%b\n\nsuff=%b\n",X,Y,rst,activeREG,cSig,counterAux,aAux3,qAux3,suff);
    a <= aAux3;
    q <= qAux3;
    counter <= counterAux;
    if(cSig[7])begin activeREG=0; suff=1; end
    quatient<= {{35{q[31]}},q};
    remainder<=a;
    rst=sec;
    sec=1;
end
endmodule











/*
module divider_tb;
  
  reg [31:0] X,Y;
  reg clk,enable;
  wire suff;
  wire [66:0] quatient;
  wire [32:0] remainder;
  
  divider divider_inst (.suff(suff), 
   .X(X),.Y(Y),.clk(clk),.active(enable),.quatient(quatient),.remainder(remainder)
  );

  // Stimulus
  initial begin
    // Initialize inputs
    //$monitor("quatient = %b\nremainder = %b\nsuff=%b\n",quatient,remainder,suff);
    //x 0 0 0 10010 1100 0010
    //y 0 0 0 00000 1010 1100
    X = 32'd4802; // Example input value
    Y = 32'd172; // Example input value
    enable=1'b1;
    // Wait some time
    #4010;

    // End simulation
    $finish;
  end
  
localparam run_cycle=10,cycles=200;
initial begin
  clk=1'b0;
  repeat (cycles*2)
#run_cycle clk=~clk;
end
  
endmodule
*/
