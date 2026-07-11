library verilog;
use verilog.vl_types.all;
entity FAC_Star is
    port(
        A               : in     vl_logic;
        B               : in     vl_logic;
        cin             : in     vl_logic;
        sum             : out    vl_logic;
        cout            : out    vl_logic;
        pi              : out    vl_logic
    );
end FAC_Star;
