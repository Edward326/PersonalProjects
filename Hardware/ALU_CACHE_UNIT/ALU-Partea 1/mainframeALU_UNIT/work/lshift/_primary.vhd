library verilog;
use verilog.vl_types.all;
entity lshift is
    port(
        c5              : in     vl_logic;
        clk             : in     vl_logic;
        a               : in     vl_logic_vector(33 downto 0);
        q               : in     vl_logic_vector(32 downto 0);
        qNeg            : in     vl_logic;
        aOUT            : out    vl_logic_vector(33 downto 0);
        qOUT            : out    vl_logic_vector(32 downto 0);
        qNegOUT         : out    vl_logic
    );
end lshift;
