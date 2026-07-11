library verilog;
use verilog.vl_types.all;
entity RCA is
    port(
        A               : in     vl_logic_vector(7 downto 0);
        B               : in     vl_logic_vector(7 downto 0);
        cin             : in     vl_logic;
        sum             : out    vl_logic_vector(7 downto 0);
        cout            : out    vl_logic;
        overflw         : out    vl_logic
    );
end RCA;
