// made and tested in EDA_PLAYGROUND environment
module controlUnit(
input clk,rst_b,
input EN,
input W_R,CacheHIT,CacheFULL,
output reg [6:0] cSig
);
localparam s0 = 1;
localparam s1 = 2;
localparam sREADINT=4;
localparam s2 = 8;
localparam s3 = 16;
localparam s4 = 32;
localparam s5 = 64;
localparam s6 = 128;
localparam s7 = 256;
localparam sWRITEINT=512;
localparam s8 = 1024;
localparam s9 = 2048;

reg [11:0] cst,nxst;

always @* begin
case(cst)
s0:begin
if(EN==1)
if(W_R==1)
nxst=s7;
else
nxst=s1;
else
nxst=s0;
end

s1:begin
nxst=sREADINT;
end

sREADINT:begin
if(CacheHIT==1)
nxst=s2;
else
nxst=s3;
end

s2:begin
nxst=s0;
end

s3:begin
if(CacheFULL==1)
nxst=s5;
else
nxst=s4;
end

s4:begin
nxst=s6;
end

s5:begin
nxst=s6;
end

s6:begin
if(W_R==1)
nxst=s7;
else
nxst=s1;
end

s7:begin
nxst=sWRITEINT;
end
  
sWRITEINT:begin
if(CacheHIT==1)
nxst=s9;
else
nxst=s8;
end

s8:begin
if(CacheFULL==1)
nxst=s5;
else
nxst=s4;
end

s9:begin
nxst=s0;
end
endcase
end

always @* begin
    case (cst)
        s0:cSig=7'd1;
        s1:cSig=7'd2;
        s2:cSig=7'd4;
        s4:cSig=7'd8;
        s5:cSig=7'd16;
        s6:cSig=7'd32;
        s7:cSig=7'd2;
        s9:cSig=7'd64;
      
        default: cSig=7'd0;
    endcase
end

always @(posedge clk,negedge rst_b) begin
//$display("q = %b", cst);
if(!rst_b)
cst<=s0;
else
cst<=nxst;       
end
endmodule

//=========================================================================>

module comparator(
  input [6:0] a,
  input [6:0] b,
  output reg eq
);
  always @* begin
    eq = (a == b) ? 1'b1 : 1'b0;
  end
endmodule

module selectWord(
input [6:0] tag,
input [6:0] index,
output reg [3:0] set,
  input [522:0] cacheBANK_A [127:0],
  input [522:0] cacheBANK_B [127:0],
  input [522:0] cacheBANK_C [127:0],
  input [522:0] cacheBANK_D [127:0]
);
wire eq1,eq2,eq3,eq4;
  comparator comp1( .a(cacheBANK_A[index][10:4]), .b(tag), .eq(eq1));
  comparator comp2( .a(cacheBANK_B[index][10:4]), .b(tag), .eq(eq2));
  comparator comp3( .a(cacheBANK_C[index][10:4]), .b(tag), .eq(eq3));
  comparator comp4( .a(cacheBANK_D[index][10:4]), .b(tag), .eq(eq4));
always @* begin
  set[0] <=eq1&cacheBANK_A[index][0];
  set[1] <=eq2&cacheBANK_B[index][0];
  set[2] <=eq3&cacheBANK_C[index][0];
  set[3] <=eq4&cacheBANK_D[index][0];
end
endmodule

//=========================================================================>

module mux4to1FULL(
    input [1:0] data_in0,
    input [1:0] data_in1,
    input [1:0] data_in2,
    input [1:0] data_in3,
    input [3:0] select,
    output reg [1:0] data_out
);
always @* begin
    case (select)
        4'd1: data_out = data_in0;
        4'd2: data_out = data_in1;
        4'd4: data_out = data_in2;
        4'd8: data_out = data_in3;
        default: data_out = 2'd0; 
    endcase
end
endmodule

module mux4to1(
    input [511:0] data_in0,
    input [511:0] data_in1,
    input [511:0] data_in2,
    input [511:0] data_in3,
    input [3:0] select,
    output reg [511:0] data_out
);
always @* begin
    case (select)
        4'd1: data_out = data_in0;
        4'd2: data_out = data_in1;
        4'd4: data_out = data_in2;
        4'd8: data_out = data_in3;
        default: data_out = 512'd0; 
    endcase
end
endmodule

module mux8to1A(
    input [63:0] data_in0,
    input [63:0] data_in1,
    input [63:0] data_in2,
    input [63:0] data_in3,
    input [63:0] data_in4,
    input [63:0] data_in5,
    input [63:0] data_in6,
    input [63:0] data_in7,
    input [2:0] select,
    output reg [63:0] data_out
);
always @* begin
    case (select)
        3'd0: data_out = data_in0;
        3'd1: data_out = data_in1;
        3'd2: data_out = data_in2;
        3'd3: data_out = data_in3;
        3'd4: data_out = data_in4;
        3'd5: data_out = data_in5;
        3'd6: data_out = data_in6;
        3'd7: data_out = data_in7;
        default: data_out = 64'd0; 
    endcase
end
endmodule

module mux8to1B(
    input [7:0] data_in0,
    input [7:0] data_in1,
    input [7:0] data_in2,
    input [7:0] data_in3,
    input [7:0] data_in4,
    input [7:0] data_in5,
    input [7:0] data_in6,
    input [7:0] data_in7,
    input [2:0] select,
    output reg [7:0] data_out
);
always @* begin
    case (select)
        3'd0: data_out = data_in0;
        3'd1: data_out = data_in1;
        3'd2: data_out = data_in2;
        3'd3: data_out = data_in3;
        3'd4: data_out = data_in4;
        3'd5: data_out = data_in5;
        3'd6: data_out = data_in6;
        3'd7: data_out = data_in7;
        default: data_out = 8'd0; 
    endcase
end
endmodule

module mux2to1(
    input [7:0] data_in0,
    input [7:0] data_in1,
    input select,
    output reg [7:0] data_out
);
always @* begin
    case (select)
        1'b0: data_out = data_in0;
        1'b1: data_out = data_in1;
        default: data_out = 8'd0; 
    endcase
end
endmodule

module mux2to1SET(
    input [1:0] data_in0,
    input [1:0] data_in1,
    input select,
    output reg [1:0] data_out
);
always @* begin
    case (select)
        1'b0: data_out = data_in0;
        1'b1: data_out = data_in1;
        default: data_out = 2'd0; 
    endcase
end
endmodule

//=========================================================================>

module eqFULLM(
input [1:0] age0,age1,age2,age3,
output reg [3:0] eqFULL
);
always @* begin
eqFULL[0]<=(age0==2'd3) ? 1'b1 : 1'b0;
eqFULL[1]<=(age1==2'd3) ? 1'b1 : 1'b0;
eqFULL[2]<=(age2==2'd3) ? 1'b1 : 1'b0;
eqFULL[3]<=(age3==2'd3) ? 1'b1 : 1'b0;
end
endmodule

module eqNFULLM(
input s0,s1,s2,s3,
output reg [3:0] eqNFULL
);
always @* begin
  eqNFULL[0]<=(~s0)&(~s1)&(~s2)&(~s3);
  eqNFULL[1]<=s0&(~s1)&(~s2)&(~s3);
  eqNFULL[2]<=s0&s1&(~s2)&(~s3);
  eqNFULL[3]<=s0&s1&s2&(~s3);
end
endmodule

//=========================================================================>

module detectBlockInCache(
input clk,
  input [522:0] cacheBANK_A [127:0],
  input [522:0] cacheBANK_B [127:0],
  input [522:0] cacheBANK_C [127:0],
  input [522:0] cacheBANK_D [127:0],
input [19:0] adressWord,
input signal,
input CacheHIT_IN,CacheFULL_IN,
input [1:0] SET_IN,
input [7:0] wordSEL_IN,
output reg CacheHIT_OUT,CacheFULL_OUT,
output reg [7:0] wordSEL_OUT,
output reg [1:0] SET_OUT
);
wire [3:0] set;
wire [511:0] selblk;
wire [63:0] selword;
wire [7:0] selbyte;

wire [7:0] wordSEL;
wire CacheHIT,CacheFULL;
wire [1:0] SET;

wire [3:0] eqFULL,eqNFULL;
wire [1:0] setFULL,setNFULL;
wire [1:0] setHIT,setNHIT;

selectWord inst(.tag(adressWord[19:13]), .index(adressWord[12:6]), .set(set),
                .cacheBANK_A(cacheBANK_A), .cacheBANK_B(cacheBANK_B), .cacheBANK_C(cacheBANK_C), .cacheBANK_D(cacheBANK_D));

mux4to1 blockSET(.data_in0(cacheBANK_A[adressWord[12:6]][522:11]), .data_in1(cacheBANK_B[adressWord[12:6]][522:11]),
                 .data_in2(cacheBANK_C[adressWord[12:6]][522:11]), .data_in3(cacheBANK_D[adressWord[12:6]][522:11]), 
                 .select(set), .data_out(selblk));
mux8to1A wordSET(.data_in0(selblk[63:0]), .data_in1(selblk[127:64]), .data_in2(selblk[191:128]),
                 .data_in3(selblk[255:192]), .data_in4(selblk[319:256]), .data_in5(selblk[383:320]),
                 .data_in6(selblk[447:384]), .data_in7(selblk[511:448]), 
                 .select(adressWord[5:3]), .data_out(selword));
mux8to1B byteSET(.data_in0(selword[7:0]), .data_in1(selword[15:8]), .data_in2(selword[23:16]),
                 .data_in3(selword[31:24]), .data_in4(selword[39:32]), .data_in5(selword[47:40]),
                 .data_in6(selword[55:48]), .data_in7(selword[63:56]), 
                 .select(adressWord[2:0]), .data_out(selbyte));  

mux2to1 byteSETFINAL(.data_in0(wordSEL_IN), .data_in1(selbyte), .select(signal), .data_out(wordSEL));
//pt wordSEL

assign CacheFULL=cacheBANK_A[adressWord[12:6]][0] & cacheBANK_B[adressWord[12:6]][0] & cacheBANK_C[adressWord[12:6]][0] & cacheBANK_D[adressWord[12:6]][0];
assign CacheHIT=(set[0] | set[1] | set[2] | set[3]);
//pt CacheHIT CacheFULL

eqFULLM eqinst1(.age0(cacheBANK_A[adressWord[12:6]][3:2]), .age1(cacheBANK_B[adressWord[12:6]][3:2]),
                  .age2(cacheBANK_C[adressWord[12:6]][3:2]), .age3(cacheBANK_D[adressWord[12:6]][3:2]), .eqFULL(eqFULL));
eqNFULLM eqinst2(.s0(cacheBANK_A[adressWord[12:6]][0]), .s1(cacheBANK_B[adressWord[12:6]][0]),
                 .s2(cacheBANK_C[adressWord[12:6]][0]), .s3(cacheBANK_D[adressWord[12:6]][0]), .eqNFULL(eqNFULL));
                
mux4to1FULL fullMUX(.data_in0(2'd0), .data_in1(2'd1), .data_in2(2'd2), .data_in3(2'd3), .select(eqFULL[3:0]), .data_out(setFULL));
mux4to1FULL nfullMUX(.data_in0(2'd0), .data_in1(2'd1), .data_in2(2'd2), .data_in3(2'd3), .select(eqNFULL[3:0]), .data_out(setNFULL));
mux2to1SET FULLORNFULLMUX(.data_in0(setNFULL), .data_in1(setFULL), .select(CacheFULL), .data_out(setNHIT));

mux4to1FULL HITCASE(.data_in0(2'd0), .data_in1(2'd1), .data_in2(2'd2), .data_in3(2'd3), .select(set[3:0]), .data_out(setHIT));
mux2to1SET HITORNHIT(.data_in0(setHIT), .data_in1(setNHIT), .select(~CacheHIT), .data_out(SET));
//pt SET

always @* begin
CacheHIT_OUT<=(signal==1)? CacheHIT:CacheHIT_IN;
CacheFULL_OUT<=(signal==1)? CacheFULL:CacheFULL_IN;
wordSEL_OUT<=wordSEL;
SET_OUT<=(signal==1)? SET:SET_IN;
end
endmodule

//==========================================================================>

module ReadHit(
  input clk, CacheFULL,signal,misscase,
  input [1:0] SET,
  input [6:0] index,
  input [7:0] wordSEL,
  input [522:0] cacheBANK_AIN [127:0],
  input [522:0] cacheBANK_BIN [127:0],
  input [522:0] cacheBANK_CIN [127:0],
  input [522:0] cacheBANK_DIN [127:0],
  output reg [522:0] cacheBANK_AOUT [127:0],
  output reg [522:0] cacheBANK_BOUT [127:0],
  output reg [522:0] cacheBANK_COUT [127:0],
  output reg [522:0] cacheBANK_DOUT [127:0]
);

reg [1:0] SETage;
integer i;

  always @* begin
  // Propagate inputs to outputs
  for (i = 0; i < 128; i = i + 1) begin
    cacheBANK_AOUT[i] = cacheBANK_AIN[i];
    cacheBANK_BOUT[i] = cacheBANK_BIN[i];
    cacheBANK_COUT[i] = cacheBANK_CIN[i];
    cacheBANK_DOUT[i] = cacheBANK_DIN[i];
  end

  if (signal) begin
    // Determine SETage based on the SET value
    case (SET)
      2'd0: SETage = cacheBANK_AOUT[index][3:2];
      2'd1: SETage = cacheBANK_BOUT[index][3:2];
      2'd2: SETage = cacheBANK_COUT[index][3:2];
      2'd3: SETage = cacheBANK_DOUT[index][3:2];
    endcase

    // Update cacheBANK_XOUT based on SETage
    if (cacheBANK_AOUT[index][3:2] < SETage && cacheBANK_AOUT[index][0] == 1)
      cacheBANK_AOUT[index][3:2] = cacheBANK_AOUT[index][3:2] + 1;
    if (cacheBANK_BOUT[index][3:2] < SETage && cacheBANK_BOUT[index][0] == 1)
      cacheBANK_BOUT[index][3:2] = cacheBANK_BOUT[index][3:2] + 1;
    if (cacheBANK_COUT[index][3:2] < SETage && cacheBANK_COUT[index][0] == 1)
      cacheBANK_COUT[index][3:2] = cacheBANK_COUT[index][3:2] + 1;
    if (cacheBANK_DOUT[index][3:2] < SETage && cacheBANK_DOUT[index][0] == 1)
      cacheBANK_DOUT[index][3:2] = cacheBANK_DOUT[index][3:2] + 1;

    // Update the selected SET
    case (SET)
      2'd0: begin
        //cacheBANK_AOUT[index][0] = 1;//(valid bit)==>automatically set when miss,or already set when hit
        cacheBANK_AOUT[index][3:2] = 2'd0;
      end
      2'd1: begin
        //cacheBANK_BOUT[index][0] = 1;
        cacheBANK_BOUT[index][3:2] = 2'd0;
      end
      2'd2: begin
        //cacheBANK_COUT[index][0] = 1;
        cacheBANK_COUT[index][3:2] = 2'd0;
      end
      2'd3: begin
        //cacheBANK_DOUT[index][0] = 1;
        cacheBANK_DOUT[index][3:2] = 2'd0;
      end
    endcase
    $display("READrequest");
    if(!misscase)
      $display("word found in cache,CACHE HIT   ==>data selected is (dec)%d",wordSEL);
    else
      $display("word not found in cache,FETCHED AFTER MISS   ==>data selected is (dec)%d=fetched_from_MM",wordSEL);
  end
end
endmodule

//==========================================================================>

module WriteHit(
  input clk, CacheFULL, signal, misscase,
  input [1:0] SET,
  input [6:0] index,
  input [2:0] blockOffset, wordOffset,
  input [7:0] wordSEL, data_in,
  input [522:0] cacheBANK_AIN [127:0],
  input [522:0] cacheBANK_BIN [127:0],
  input [522:0] cacheBANK_CIN [127:0],
  input [522:0] cacheBANK_DIN [127:0],
  output reg [522:0] cacheBANK_AOUT [127:0],
  output reg [522:0] cacheBANK_BOUT [127:0],
  output reg [522:0] cacheBANK_COUT [127:0],
  output reg [522:0] cacheBANK_DOUT [127:0]
);

reg [1:0] SETage;
integer i;
integer byte_pos;

always @* begin
  // Propagate inputs to outputs
  for (i = 0; i < 128; i = i + 1) begin
    cacheBANK_AOUT[i] = cacheBANK_AIN[i];
    cacheBANK_BOUT[i] = cacheBANK_BIN[i];
    cacheBANK_COUT[i] = cacheBANK_CIN[i];
    cacheBANK_DOUT[i] = cacheBANK_DIN[i];
  end

  if (signal) begin
    // Determine SETage based on the SET value
    case (SET)
      2'd0: SETage = cacheBANK_AOUT[index][3:2];
      2'd1: SETage = cacheBANK_BOUT[index][3:2];
      2'd2: SETage = cacheBANK_COUT[index][3:2];
      2'd3: SETage = cacheBANK_DOUT[index][3:2];
    endcase

    // Update cacheBANK_XOUT based on SETage
    if (cacheBANK_AOUT[index][3:2] < SETage && cacheBANK_AOUT[index][0] == 1)
      cacheBANK_AOUT[index][3:2] = cacheBANK_AOUT[index][3:2] + 1;
    if (cacheBANK_BOUT[index][3:2] < SETage && cacheBANK_BOUT[index][0] == 1)
      cacheBANK_BOUT[index][3:2] = cacheBANK_BOUT[index][3:2] + 1;
    if (cacheBANK_COUT[index][3:2] < SETage && cacheBANK_COUT[index][0] == 1)
      cacheBANK_COUT[index][3:2] = cacheBANK_COUT[index][3:2] + 1;
    if (cacheBANK_DOUT[index][3:2] < SETage && cacheBANK_DOUT[index][0] == 1)
      cacheBANK_DOUT[index][3:2] = cacheBANK_DOUT[index][3:2] + 1;

    // Calculate byte position within the cache line
    byte_pos = blockOffset * 64 + wordOffset * 8;

    // Update the selected SET
    case (SET)
      2'd0: begin
        //cacheBANK_AOUT[index][0] = 1;//(valid bit)==>automatically set when miss,or already set when hit
        cacheBANK_AOUT[index][3:2] = 2'd0;
        cacheBANK_AOUT[index][1] = (wordSEL != data_in) ? 1'b1 : 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
          cacheBANK_AOUT[index][11+byte_pos + i] = (wordSEL != data_in) ? data_in[i] : wordSEL[i];
        end
      end
      2'd1: begin
        //cacheBANK_BOUT[index][0] = 1;
        cacheBANK_BOUT[index][3:2] = 2'd0;
        cacheBANK_BOUT[index][1] = (wordSEL != data_in) ? 1'b1 : 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
          cacheBANK_BOUT[index][11+byte_pos + i] = (wordSEL != data_in) ? data_in[i] : wordSEL[i];
        end
      end
      2'd2: begin
        //cacheBANK_COUT[index][0] = 1;
        cacheBANK_COUT[index][3:2] = 2'd0;
        cacheBANK_COUT[index][1] = (wordSEL != data_in) ? 1'b1 : 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
          cacheBANK_COUT[index][11+byte_pos + i] = (wordSEL != data_in) ? data_in[i] : wordSEL[i];
        end
      end
      2'd3: begin
        //cacheBANK_DOUT[index][0] = 1;
        cacheBANK_DOUT[index][3:2] = 2'd0;
        cacheBANK_DOUT[index][1] = (wordSEL != data_in) ? 1'b1 : 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
          cacheBANK_DOUT[index][11+byte_pos + i] = (wordSEL != data_in) ? data_in[i] : wordSEL[i];
        end
      end
    endcase
    $display("WRITErequest");
    if(!misscase)
      $display("word found in cache,CACHE HIT   ==>data that will be written in, is (dec)%d=data_in",data_in);
    else
      $display("word not found in cache,FETCHED AFTER MISS   ==>data that will be written in, is (dec)%d=data_in",data_in);
  end
end

endmodule

//==============================================================>

module MissCaseNFULL(
  input clk,signal,
  input [1:0] SET,
  input [6:0] index,
  input [522:0] cacheBANK_AIN [127:0],
  input [522:0] cacheBANK_BIN [127:0],
  input [522:0] cacheBANK_CIN [127:0],
  input [522:0] cacheBANK_DIN [127:0],
  output reg [522:0] cacheBANK_AOUT [127:0],
  output reg [522:0] cacheBANK_BOUT [127:0],
  output reg [522:0] cacheBANK_COUT [127:0],
  output reg [522:0] cacheBANK_DOUT [127:0]
);

integer i;

  always @* begin
  // Propagate inputs to outputs
  for (i = 0; i < 128; i = i + 1) begin
    cacheBANK_AOUT[i] = cacheBANK_AIN[i];
    cacheBANK_BOUT[i] = cacheBANK_BIN[i];
    cacheBANK_COUT[i] = cacheBANK_CIN[i];
    cacheBANK_DOUT[i] = cacheBANK_DIN[i];
  end

  if (signal) begin
    for (i = 0; i <SET; i = i + 1) begin
      if(i==0)begin
        cacheBANK_AOUT[index][3:2] = cacheBANK_AOUT[index][3:2] + 1;
      end
      if(i==1)begin
        cacheBANK_BOUT[index][3:2] = cacheBANK_BOUT[index][3:2] + 1;
      end
      if(i==2)begin
        cacheBANK_COUT[index][3:2] = cacheBANK_COUT[index][3:2] + 1;
      end
     end

    // Update the selected SET
    case (SET)
      2'd0: begin
        //cacheBANK_AOUT[index][0] = 1;//(valid bit)==>will be setted in Allocate fct
        cacheBANK_AOUT[index][3:2] = 2'd0;
      end
      2'd1: begin
        //cacheBANK_BOUT[index][0] = 1;
        cacheBANK_BOUT[index][3:2] = 2'd0;
      end
      2'd2: begin
        //cacheBANK_COUT[index][0] = 1;
        cacheBANK_COUT[index][3:2] = 2'd0;
      end
      2'd3: begin
        //cacheBANK_DOUT[index][0] = 1;
        cacheBANK_DOUT[index][3:2] = 2'd0;
      end
    endcase
  end
end
endmodule

//==========================================================================>

module MissCaseFULL(
  input clk,signal,
  input [1:0] SET,
  input [6:0] index,
  input [522:0] cacheBANK_AIN [127:0],
  input [522:0] cacheBANK_BIN [127:0],
  input [522:0] cacheBANK_CIN [127:0],
  input [522:0] cacheBANK_DIN [127:0],
  output reg [522:0] cacheBANK_AOUT [127:0],
  output reg [522:0] cacheBANK_BOUT [127:0],
  output reg [522:0] cacheBANK_COUT [127:0],
  output reg [522:0] cacheBANK_DOUT [127:0]
);

integer i;

  always @* begin
  // Propagate inputs to outputs
  for (i = 0; i < 128; i = i + 1) begin
    cacheBANK_AOUT[i] = cacheBANK_AIN[i];
    cacheBANK_BOUT[i] = cacheBANK_BIN[i];
    cacheBANK_COUT[i] = cacheBANK_CIN[i];
    cacheBANK_DOUT[i] = cacheBANK_DIN[i];
  end

  if (signal) begin
    
    cacheBANK_AOUT[index][3:2] = cacheBANK_AOUT[index][3:2] + 1;
    cacheBANK_BOUT[index][3:2] = cacheBANK_BOUT[index][3:2] + 1;
    cacheBANK_COUT[index][3:2] = cacheBANK_COUT[index][3:2] + 1;
    cacheBANK_DOUT[index][3:2] = cacheBANK_DOUT[index][3:2] + 1;
    
    // Update the selected SET
    case (SET)
      2'd0: begin
        //cacheBANK_AOUT[index][0] = 1;//(valid bit)==>will be setted in Allocate fct
        cacheBANK_AOUT[index][3:2] = 2'd0;
      end
      2'd1: begin
        //cacheBANK_BOUT[index][0] = 1;
        cacheBANK_BOUT[index][3:2] = 2'd0;
      end
      2'd2: begin
        //cacheBANK_COUT[index][0] = 1;
        cacheBANK_COUT[index][3:2] = 2'd0;
      end
      2'd3: begin
        //cacheBANK_DOUT[index][0] = 1;
        cacheBANK_DOUT[index][3:2] = 2'd0;
      end
    endcase
  end
end
endmodule

//==========================================================================>

module Allocate(
  input clk, signal,
  input [1:0] SET,
  input [6:0] index, tag,
  input [522:0] cacheBANK_AIN [127:0],
  input [522:0] cacheBANK_BIN [127:0],
  input [522:0] cacheBANK_CIN [127:0],
  input [522:0] cacheBANK_DIN [127:0],
  output reg [522:0] cacheBANK_AOUT [127:0],
  output reg [522:0] cacheBANK_BOUT [127:0],
  output reg [522:0] cacheBANK_COUT [127:0],
  output reg [522:0] cacheBANK_DOUT [127:0]
);

integer i;
reg [511:0] random_data;
reg [7:0] random_byte;
  
always @* begin
  // Propagate inputs to outputs
  for (i = 0; i < 128; i = i + 1) begin
    cacheBANK_AOUT[i] = cacheBANK_AIN[i];
    cacheBANK_BOUT[i] = cacheBANK_BIN[i];
    cacheBANK_COUT[i] = cacheBANK_CIN[i];
    cacheBANK_DOUT[i] = cacheBANK_DIN[i];
  end
  
  if (signal) begin
    random_data=512'd0;
    // Generam 8 cuvinte fiecare retinand 8 nr(pe8b=1B)
      for (i = 0; i < 64; i = i + 1) begin
      random_byte = $urandom % 256; // Generate an 8-bit random number
      random_data[i*8 +: 8] = random_byte; // Assign 8 bits at a time
      end 

    // Update the selected SET
    case (SET)
      2'd0: begin
        cacheBANK_AOUT[index][0] = 1;
        cacheBANK_AOUT[index][10:4] = tag;
        cacheBANK_AOUT[index][522:11] = random_data;
      end
      2'd1: begin
        cacheBANK_BOUT[index][0] = 1;
        cacheBANK_BOUT[index][10:4] = tag;
        cacheBANK_BOUT[index][522:11] = random_data;
      end
      2'd2: begin
        cacheBANK_COUT[index][0] = 1;
        cacheBANK_COUT[index][10:4] = tag;
        cacheBANK_COUT[index][522:11] = random_data;
      end
      2'd3: begin
        cacheBANK_DOUT[index][0] = 1;
        cacheBANK_DOUT[index][10:4] = tag;
        cacheBANK_DOUT[index][522:11] = random_data;
      end
    endcase
  end
end

endmodule

//==========================================================================>

//CACHE CENTRAL UNIT
module centralUnit(
input [19:0] adressWord,
input [7:0] data_in,
input clk,
input EN,W_R,
output reg [7:0] wordSELOUT
);

//4 matrices of dimension:128 lines with 524 columns(all fields from one banks's cache index/line/no_set)
  reg [522:0] cacheBANK_A [127:0];
  reg [522:0] cacheBANK_B [127:0];
  reg [522:0] cacheBANK_C [127:0];
  reg [522:0] cacheBANK_D [127:0];
  
  wire [522:0] cacheBANK_Aaux [127:0];
  wire [522:0] cacheBANK_Baux [127:0];
  wire [522:0] cacheBANK_Caux [127:0];
  wire [522:0] cacheBANK_Daux [127:0];

  wire [522:0] cacheBANK_Aaux2 [127:0];
  wire [522:0] cacheBANK_Baux2 [127:0];
  wire [522:0] cacheBANK_Caux2 [127:0];
  wire [522:0] cacheBANK_Daux2 [127:0];

  wire [522:0] cacheBANK_Aaux3 [127:0];
  wire [522:0] cacheBANK_Baux3 [127:0];
  wire [522:0] cacheBANK_Caux3 [127:0];
  wire [522:0] cacheBANK_Daux3 [127:0];
  
  wire [522:0] cacheBANK_Aaux4 [127:0];
  wire [522:0] cacheBANK_Baux4 [127:0];
  wire [522:0] cacheBANK_Caux4 [127:0];
  wire [522:0] cacheBANK_Daux4 [127:0];
  
  wire [522:0] cacheBANK_Aaux5 [127:0];
  wire [522:0] cacheBANK_Baux5 [127:0];
  wire [522:0] cacheBANK_Caux5 [127:0];
  wire [522:0] cacheBANK_Daux5 [127:0];

reg CacheHIT,CacheFULL;
reg [1:0] SET;
reg [7:0] wordSEL;
reg misscasereg;

wire [6:0] cSig;
wire CacheHITaux,CacheFULLaux;
wire [1:0] SETaux;
wire [7:0] wordSELaux;
reg rst=0,sec=0;

integer i, j;
  always @(posedge clk) begin
    if(!rst) begin
     wordSEL<=8'd0;CacheHIT<=0;CacheFULL<=0;SET<=2'd0;misscasereg<=1'b0;
  for (i = 0; i < 128; i = i + 1) begin
      cacheBANK_A[i]<=524'd0;cacheBANK_B[i]<=524'd0;cacheBANK_C[i]<=524'd0;cacheBANK_D[i]<=524'd0;
  end
      /*
      //cache test for line1 word8 wordOffB=0 ,for HIT CASE from start
      //cacheBANK_D[1][1]<=1'b1;cacheBANK_D[1][523:460]<=64'hFFFFFFFFFFFFFFFF;cacheBANK_D[1][4:3]<=2'd2;
      //cacheBANK_A[1][0]<=1'b0;cacheBANK_B[1][0]<=1'b0;cacheBANK_C[1][0]<=1'b0;cacheBANK_D[1][0]<=1'b0;
      */
    end
  end

controlUnit fsm(.clk(clk), .rst_b(rst), .EN(EN), 
                .W_R(W_R), .CacheHIT(CacheHIT), 
                .CacheFULL(CacheFULL), .cSig(cSig));
  
detectBlockInCache detect(.clk(clk), .adressWord(adressWord), .signal(cSig[1]),
                          .cacheBANK_A(cacheBANK_A), .cacheBANK_B(cacheBANK_B), 
                          .cacheBANK_C(cacheBANK_C), .cacheBANK_D(cacheBANK_D),
                          .CacheHIT_IN(CacheHIT), .wordSEL_IN(wordSEL), 
                          .CacheFULL_IN(CacheFULL), .SET_IN(SET),
                          .CacheHIT_OUT(CacheHITaux), .wordSEL_OUT(wordSELaux), 
                          .CacheFULL_OUT(CacheFULLaux), .SET_OUT(SETaux));
  
ReadHit hitread(.clk(clk), .CacheFULL(CacheFULLaux), .SET(SETaux), .signal(cSig[2]), .misscase(misscasereg),
                .index(adressWord[12:6]), .wordSEL(wordSELaux), 
                .cacheBANK_AIN(cacheBANK_A), .cacheBANK_BIN(cacheBANK_B), 
                .cacheBANK_CIN(cacheBANK_C), .cacheBANK_DIN(cacheBANK_D),
                .cacheBANK_AOUT(cacheBANK_Aaux), .cacheBANK_BOUT(cacheBANK_Baux),
                .cacheBANK_COUT(cacheBANK_Caux), .cacheBANK_DOUT(cacheBANK_Daux));
  
MissCaseNFULL misscasenfull(.clk(clk), .SET(SETaux), .signal(cSig[3]), 
                            .index(adressWord[12:6]),  
                            .cacheBANK_AIN(cacheBANK_Aaux), .cacheBANK_BIN(cacheBANK_Baux),
                            .cacheBANK_CIN(cacheBANK_Caux), .cacheBANK_DIN(cacheBANK_Daux),
                            .cacheBANK_AOUT(cacheBANK_Aaux2), .cacheBANK_BOUT(cacheBANK_Baux2),
                            .cacheBANK_COUT(cacheBANK_Caux2), .cacheBANK_DOUT(cacheBANK_Daux2));

MissCaseFULL misscasefull(.clk(clk), .SET(SETaux), .signal(cSig[4]), 
                          .index(adressWord[12:6]),  
                          .cacheBANK_AIN(cacheBANK_Aaux2), .cacheBANK_BIN(cacheBANK_Baux2),
                          .cacheBANK_CIN(cacheBANK_Caux2), .cacheBANK_DIN(cacheBANK_Daux2),
                          .cacheBANK_AOUT(cacheBANK_Aaux3), .cacheBANK_BOUT(cacheBANK_Baux3),
                          .cacheBANK_COUT(cacheBANK_Caux3), .cacheBANK_DOUT(cacheBANK_Daux3));
  
Allocate alloc(.clk(clk), .SET(SETaux), .signal(cSig[5]),
               .index(adressWord[12:6]), .tag(adressWord[19:13]),
               .cacheBANK_AIN(cacheBANK_Aaux3), .cacheBANK_BIN(cacheBANK_Baux3),
               .cacheBANK_CIN(cacheBANK_Caux3), .cacheBANK_DIN(cacheBANK_Daux3),
               .cacheBANK_AOUT(cacheBANK_Aaux4), .cacheBANK_BOUT(cacheBANK_Baux4),
               .cacheBANK_COUT(cacheBANK_Caux4), .cacheBANK_DOUT(cacheBANK_Daux4));
  
WriteHit hitwrite(.clk(clk), .CacheFULL(CacheFULLaux), .SET(SETaux), .signal(cSig[6]), .misscase(misscasereg),
                  .index(adressWord[12:6]), .wordSEL(wordSELaux), .data_in(data_in),
                  .blockOffset(adressWord[5:3]), .wordOffset(adressWord[2:0]),
                  .cacheBANK_AIN(cacheBANK_Aaux4), .cacheBANK_BIN(cacheBANK_Baux4), 
                  .cacheBANK_CIN(cacheBANK_Caux4), .cacheBANK_DIN(cacheBANK_Daux4),
                  .cacheBANK_AOUT(cacheBANK_Aaux5), .cacheBANK_BOUT(cacheBANK_Baux5),
                  .cacheBANK_COUT(cacheBANK_Caux5), .cacheBANK_DOUT(cacheBANK_Daux5));

   always @(posedge clk) begin
     //pt a monitoriza iesirile,cazult de hit,bank-ul selectat,cazul daca cach-ul pe acea linie e full sau nu
     //$monitor("rst=%b  cSig=%b  HIT=%b  wordSEL=%d  SET=%b  FULL=%b",rst,cSig,CacheHITaux,wordSELaux,SETaux,CacheFULLaux);
    for (i = 0; i < 128; i = i + 1) begin
      cacheBANK_A[i] <= cacheBANK_Aaux5[i];
      cacheBANK_B[i] <= cacheBANK_Baux5[i];
      cacheBANK_C[i] <= cacheBANK_Caux5[i];
      cacheBANK_D[i] <= cacheBANK_Daux5[i];
    end
    
    if(cSig[6] | cSig[2] | cSig[0])begin
    wordSEL<=8'd0; CacheHIT<=0;CacheFULL<=0;SET<=2'd0;//nu reinit niciun bank de cache
    end
    else begin
    CacheHIT<=CacheHITaux;
    CacheFULL<=CacheFULLaux;
    SET<=SETaux;
    wordSEL<=wordSELaux;
    wordSELOUT<=wordSEL;
    end
     
    if(cSig[5])misscasereg<=1;
    if(cSig[0])misscasereg<=0;
    
    rst=sec;
    sec=1;
end
endmodule











/*
testul ar trebuii sa furnizeze:
set:
0(M),1(M),2(M),3(M),1(H),0(M),2(M),3(Hpentru scriere),3(Hpentru citire),1(M)
*/
module centralUnit_tb;
  reg [19:0] adressWord;
  reg [7:0] data_in;
  reg clk;
  reg EN,W_R;
  wire [7:0] wordSELOUT;
  
  centralUnit test(.adressWord(adressWord), .data_in(data_in),
                   .clk(clk), .EN(EN), .W_R(W_R), .wordSELOUT(wordSELOUT));
  initial begin
    //$monitor("%b\n",wordSELOUT);
    //details/specs:
    //minclocksRST_EN=2cc  ,min cc after EN is disabled for 1 request,to block the fsm in the init state,after solving the request
    //avgclocksADDRESSHIT=5cc  ,avg cc when CACHEHIT
    //avgclocksADRESSMISS=10cc  ,avg cc when CACHEMISS(ccForGoingIntoFetchState(s6) + missPenalty + ccForCacheHIT + (opt)BusWidthDelays)
    //1cc=2(levels)*10ns(time on each level)
    
    
    
    
    //test requests
    
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching
    adressWord=20'b01000000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching
    adressWord=20'b00100000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching
    adressWord=20'b10000000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching
    adressWord=20'b00010000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n\n\n\n\nLRU CASE\nwhen cache its full on the line requested ,and needs to ovwrt the data over the oldest bank\\n\n");
    
    //1request -read ==>HIT=found in cacheBANK_B(bankNo.2)
    adressWord=20'b00100000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching ,fetching now over the oldest bank(bankNo.1 because age=3) 
    adressWord=20'b11100000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request -read ==>MISS=fetching ,fetching now over the oldest bank(bankNo.3 because age=3) 
    adressWord=20'b11111000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request  -write ==>HIT=found in cacheBANK_D(bankNo.4)
    adressWord=20'b00010000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b1;data_in=8'd72;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request  -read ==>HIT=found in cacheBANK_D(bankNo.4)
    adressWord=20'b00010000000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    $display("\n\n");
    
    //1request  -read ==>MISS=fetching ,fetching now over the oldest bank(bankNo.1 because age=3) 
    adressWord=20'b11111100000001111000;//linia/index=1 word=8 wordOffB=0
    EN=1'b1;
    W_R=1'b0;
    #50;//wait minclocksRST_EN cycles from start,for fsm to exit s0 state(independent on EN)     =2cc x 20ns +10buff
    EN=1'b0;//then we rest the en until the next request
    #210;//the cache will be quicker thatn this waitTime,causing it to wait for a next request  =(miss)10cc x 20ns +10 buff
    
    
    
    
    //another requests...
    
    
    
    $display("\n\n");
    $finish;
  end
  
  localparam run_cycle=10,cycles=1000;
initial begin
  clk=1'b0;
  repeat (cycles*2)
#run_cycle clk=~clk;
end
endmodule
