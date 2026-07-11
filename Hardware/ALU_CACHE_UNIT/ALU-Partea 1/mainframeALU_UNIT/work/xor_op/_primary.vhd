library verilog;
use verilog.vl_types.all;
entity xor_op is
    port(
        a               : in     vl_logic_vector(31 downto 0);
        b               : in     vl_logic_vector(31 downto 0);
        c               : out    vl_logic_vector(66 downto 0)
    );
end xor_op;
