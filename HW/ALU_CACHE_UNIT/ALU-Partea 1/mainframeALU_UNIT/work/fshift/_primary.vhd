library verilog;
use verilog.vl_types.all;
entity fshift is
    port(
        clk             : in     vl_logic;
        active          : in     vl_logic;
        a               : in     vl_logic_vector(33 downto 0);
        q               : in     vl_logic_vector(32 downto 0);
        newa            : out    vl_logic_vector(33 downto 0);
        newq            : out    vl_logic_vector(32 downto 0)
    );
end fshift;
