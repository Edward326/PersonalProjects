library verilog;
use verilog.vl_types.all;
entity operations2 is
    port(
        clk             : in     vl_logic;
        m               : in     vl_logic_vector(32 downto 0);
        a               : in     vl_logic_vector(32 downto 0);
        cSig            : in     vl_logic_vector(1 downto 0);
        newa            : out    vl_logic_vector(32 downto 0)
    );
end operations2;
