library verilog;
use verilog.vl_types.all;
entity mux11to1 is
    port(
        data_in0        : in     vl_logic_vector(66 downto 0);
        data_in1        : in     vl_logic_vector(66 downto 0);
        data_in2        : in     vl_logic_vector(66 downto 0);
        data_in3        : in     vl_logic_vector(66 downto 0);
        data_in4        : in     vl_logic_vector(66 downto 0);
        data_in5        : in     vl_logic_vector(66 downto 0);
        data_in6        : in     vl_logic_vector(66 downto 0);
        data_in7        : in     vl_logic_vector(66 downto 0);
        data_in8        : in     vl_logic_vector(66 downto 0);
        data_in9        : in     vl_logic_vector(66 downto 0);
        data_in10       : in     vl_logic_vector(66 downto 0);
        \select\        : in     vl_logic_vector(4 downto 0);
        suff            : in     vl_logic_vector(3 downto 0);
        data_out        : out    vl_logic_vector(66 downto 0)
    );
end mux11to1;
