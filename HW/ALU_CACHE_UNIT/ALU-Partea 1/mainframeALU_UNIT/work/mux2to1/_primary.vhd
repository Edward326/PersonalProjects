library verilog;
use verilog.vl_types.all;
entity mux2to1 is
    port(
        suff            : in     vl_logic;
        data_in0        : in     vl_logic_vector(32 downto 0);
        data_in1        : in     vl_logic_vector(32 downto 0);
        \select\        : in     vl_logic;
        data_out        : out    vl_logic_vector(32 downto 0)
    );
end mux2to1;
