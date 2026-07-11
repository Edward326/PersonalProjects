library verilog;
use verilog.vl_types.all;
entity logicOp is
    port(
        X               : in     vl_logic_vector(31 downto 0);
        Y               : in     vl_logic_vector(31 downto 0);
        shiftedRX       : out    vl_logic_vector(66 downto 0);
        shiftedLX       : out    vl_logic_vector(66 downto 0);
        shiftedRY       : out    vl_logic_vector(66 downto 0);
        shiftedLY       : out    vl_logic_vector(66 downto 0);
        andOp           : out    vl_logic_vector(66 downto 0);
        orOp            : out    vl_logic_vector(66 downto 0);
        xorOp           : out    vl_logic_vector(66 downto 0);
        suff            : out    vl_logic
    );
end logicOp;
