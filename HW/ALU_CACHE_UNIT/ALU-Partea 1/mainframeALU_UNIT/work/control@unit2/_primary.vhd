library verilog;
use verilog.vl_types.all;
entity controlUnit2 is
    port(
        clk             : in     vl_logic;
        rst_b           : in     vl_logic;
        START           : in     vl_logic;
        cnt             : in     vl_logic_vector(4 downto 0);
        w               : in     vl_logic;
        cSig            : out    vl_logic_vector(7 downto 0)
    );
end controlUnit2;
