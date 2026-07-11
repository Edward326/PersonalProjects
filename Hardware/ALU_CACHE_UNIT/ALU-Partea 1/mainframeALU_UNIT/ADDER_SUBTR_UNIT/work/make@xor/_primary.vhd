library verilog;
use verilog.vl_types.all;
entity makeXor is
    generic(
        WIDTH           : integer := 32
    );
    port(
        a               : in     vl_logic_vector;
        b               : in     vl_logic;
        aXor            : out    vl_logic_vector
    );
end makeXor;
