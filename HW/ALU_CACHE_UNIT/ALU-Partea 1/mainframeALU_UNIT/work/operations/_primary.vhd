library verilog;
use verilog.vl_types.all;
entity operations is
    port(
        c7              : in     vl_logic;
        clk             : in     vl_logic;
        m               : in     vl_logic_vector(33 downto 0);
        a               : in     vl_logic_vector(33 downto 0);
        cSig            : in     vl_logic_vector(2 downto 0);
        newa            : out    vl_logic_vector(33 downto 0)
    );
end operations;
