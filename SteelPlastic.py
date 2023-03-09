import pandas as pd
import numpy as np
df = pd.read_csv(r"SteelMaterialProperties.csv")
d = 12.8524 #inch to mm
r = d/2
A = np.pi*r**2
force_steel = df['FORCE']*0.45359237*9.80665
stress_steel = (force_steel/A)*1000000 #Pa
strain_steel = df['CH5']*0.01  #Unitless in %, so 1% = 0.01
dt = pd.DataFrame({'stress':stress_steel, 'strain':strain_steel})
dt.to_csv('Stress_Strain.csv', index=False)
