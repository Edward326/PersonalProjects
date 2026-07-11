library verilog;
use verilog.vl_types.all;
entity mux4to1 is
    port(
        data_in0        : in     vl_logic_vector(33 downto 0);
        data_in1        : in     vl_logic_vector(33 downto 0);
        data_in2        : in     vl_logic_vector(33 downto 0);
        data_in3        : in     vl_logic_vector(33 downto 0);
        \select\        : in     vl_logic_vector(1 downto 0);
        data_out        : out    vl_logic_vector(33 downto 0)
    );
end mux4to1;
