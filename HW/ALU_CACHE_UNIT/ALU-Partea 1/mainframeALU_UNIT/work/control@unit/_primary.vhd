library verilog;
use verilog.vl_types.all;
entity controlUnit is
    port(
        clk             : in     vl_logic;
        rst_b           : in     vl_logic;
        START           : in     vl_logic;
        cnt             : in     vl_logic_vector(2 downto 0);
        w               : in     vl_logic;
        x               : in     vl_logic;
        y               : in     vl_logic;
        z               : in     vl_logic;
        cSig            : out    vl_logic_vector(7 downto 0)
    );
end controlUnit;
