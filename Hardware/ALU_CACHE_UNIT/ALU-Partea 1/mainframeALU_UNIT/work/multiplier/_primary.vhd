library verilog;
use verilog.vl_types.all;
entity multiplier is
    port(
        X               : in     vl_logic_vector(31 downto 0);
        Y               : in     vl_logic_vector(31 downto 0);
        clk             : in     vl_logic;
        active          : in     vl_logic;
        product         : out    vl_logic_vector(66 downto 0);
        suff            : out    vl_logic
    );
end multiplier;
