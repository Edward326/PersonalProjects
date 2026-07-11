library verilog;
use verilog.vl_types.all;
entity rshift is
    port(
        c6              : in     vl_logic;
        clk             : in     vl_logic;
        a               : in     vl_logic_vector(32 downto 0);
        q               : in     vl_logic_vector(31 downto 0);
        aOUT            : out    vl_logic_vector(32 downto 0);
        qOUT            : out    vl_logic_vector(31 downto 0)
    );
end rshift;
