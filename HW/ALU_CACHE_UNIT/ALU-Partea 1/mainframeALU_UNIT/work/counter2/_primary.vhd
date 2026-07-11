library verilog;
use verilog.vl_types.all;
entity counter2 is
    port(
        clk             : in     vl_logic;
        c_up            : in     vl_logic;
        rst             : in     vl_logic;
        clr             : in     vl_logic;
        count_reg       : in     vl_logic_vector(4 downto 0);
        count           : out    vl_logic_vector(4 downto 0)
    );
end counter2;
