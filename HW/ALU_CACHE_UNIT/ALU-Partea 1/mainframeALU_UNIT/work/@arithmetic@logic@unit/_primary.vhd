library verilog;
use verilog.vl_types.all;
entity ArithmeticLogicUnit is
    port(
        clk             : in     vl_logic;
        op              : in     vl_logic_vector(4 downto 0);
        X               : in     vl_logic_vector(31 downto 0);
        Y               : in     vl_logic_vector(31 downto 0);
        result          : out    vl_logic_vector(66 downto 0);
        remainder       : out    vl_logic_vector(32 downto 0)
    );
end ArithmeticLogicUnit;
