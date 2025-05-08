library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity LabelEncoder is
  Port ( clk : in STD_LOGIC;
         name_in : in STD_LOGIC_VECTOR(127 downto 0);
         label_out : out STD_LOGIC_VECTOR(127 downto 0));
end LabelEncoder;

architecture Behavioral of LabelEncoder is
begin
  process(clk)
  begin
    if rising_edge(clk) then
      label_out <= name_in xor x"CAFEBABE";  -- XOR encoding for uniqueness
    end if;
  end process;
end Behavioral;
