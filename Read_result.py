import re

# Open the file and read the content
with open('.\Exp_multiarm\AB_Test\AB_C1_K5_D2alpha0.05beta0.5_AB_Pareto_set_size3_BaseMean10_DeltaMean0_DeltaVar0.2_Min0.052023-12-28 110734.284912', 'r') as f:
    content = f.read()

# Use a regular expression to find the number after "Power="
match = re.search(r'Power=([\d\.]+)', content)

if match:
    power = float(match.group(1))
    print(power)
else:
    print("No match found.")