import pandas as pd
import matplotlib.pyplot as plt

archetype = 'smrf_6_of_IV'

data_dir = f'analysis/{archetype}/pushover'

df = pd.read_csv(f'{data_dir}/curve.csv', index_col=0)
df_m = pd.read_csv(f'{data_dir}/metadata.csv', index_col=0)

# Plot pushover curve ~ normalized ~ read yield drift
plt.plot(df['displ']/float(df_m['height (in)']), df['force']/1e3, 'k')
plt.xlabel('Drift ratio')
plt.ylabel('Base Shear (kips)')
plt.grid()
plt.show()


# print base shear in kips
for f in df['force'] / 1e3:
    print(f)

# print roof displacement in ft
for f in df['displ'] / 12.00:
    print(f)

# weight in kips
float(df_m['weight (lb)'])/1e3

# height in ft
float(df_m['height (in)']/12.00)


