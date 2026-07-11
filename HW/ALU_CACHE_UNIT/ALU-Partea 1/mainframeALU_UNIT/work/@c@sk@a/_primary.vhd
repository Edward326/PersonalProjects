library verilog;
use verilog.vl_types.all;
entity CSkA is
    port(
        X               : in     vl_logic_vector(31 downto 0);
        Y               : in     vl_logic_vector(31 downto 0);
        X2              : in     vl_logic_vector(32 downto 0);
        Y2              : in     vl_logic_vector(32 downto 0);
        X3              : in     vl_logic_vector(33 downto 0);
        Y3              : in     vl_logic_vector(33 downto 0);
        cin             : in     vl_logic;
        cout            : out    vl_logic;
        cout2           : out    vl_logic;
        cout3           : out    vl_logic;
        sum             : out    vl_logic_vector(66 downto 0);
        sum2            : out    vl_logic_vector(32 downto 0);
        sum3            : out    vl_logic_vector(33 downto 0);
        suff            : out    vl_logic
    );
end CSkA;
