library verilog;
use verilog.vl_types.all;
entity checkNotRestore is
    port(
        clk             : in     vl_logic;
        c5              : in     vl_logic;
        q               : in     vl_logic_vector(31 downto 0);
        newq            : out    vl_logic_vector(31 downto 0)
    );
end checkNotRestore;
