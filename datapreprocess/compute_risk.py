import geopandas as gpd  # Library for working with geospatial data
import pandas as pd  # Library for data analysis and manipulation
import matplotlib.pyplot as plt  # Library for creating visualizations
from matplotlib.gridspec import GridSpec  # Layout manager for subplots


def normalize(values):
  """
  This function normalizes a pandas Series between 0 and 1.

  Args:
      values: A pandas Series containing the values to be normalized.

  Returns:
      A new pandas Series with the normalized values.
  """
  normed = (values - values.min()) / (values.max() - values.min())
  return normed


# Read NYC geospatial data (boundaries)
gdf = gpd.read_file('MODZCTA_2010.shp')

# Ensure MODZCTA column is integer type for efficient merging
gdf['MODZCTA'] = gdf['MODZCTA'].astype('int64')

# Read NYC heat data with MODZCTA as index
data = pd.read_csv('nyc_heat_data_v2.csv', index_col='MODZCTA')

# Define data categories
hazard = ['t2', 'case_rate']  # Hazard indicators (temperature, case rate)
vulnerable = ['P_BelPov_e', 'MedInc_e', 'P_Abv65_e',  # Vulnerability indicators
             'P_Crwdd_e', 'P_NoHIns_e', 'P_POC_e']
exposure = ['TotPop_e']  # Exposure indicator (population)

# Define labels for data columns with formatting
labels = {'t2': u'Temperature (\u00b0C)',  # Temperature with degree symbol
          'case_rate': 'Case rate',
          'P_BelPov_e': '% Below pov.',
          'MedInc_e': 'Median Income',
          'P_Abv65_e': '% Above 65',
          'P_Crwdd_e': '% Overcrowded',
          'P_NoHIns_e': '% No health insurance',
          'P_POC_e': '% POC',
          'TotPop_e': 'Population'}

# Calculate case rate per capita
data['case_rate'] = data['COVID_CASE_COUNT'] / data['TotPop_e']

# Normalize all exposure, hazard, and vulnerability data (0 to 1)
for indicator in (exposure, hazard, vulnerable):
  for col in indicator:
    data[col + 'norm'] = normalize(data[col])

# Invert normalized Median Income (higher income translates to lower risk)
data['InvMedInc_enorm'] = (data['MedInc_enorm'] - 1).abs()

# Join heat data with geospatial boundaries using MODZCTA
gdf = gdf.merge(data, left_on='MODZCTA', right_index=True)

# Calculate hazard index (average of normalized temp and case rate)
gdf['hazard'] = (gdf['t2norm'] + gdf['case_rate']) / 2

# Define new vulnerability indicators (normalized columns)
newvul = ['TotPop_enorm', 'P_BelPov_enorm', 'InvMedInc_enorm', 'P_Abv65_enorm',
          'P_Crwdd_enorm', 'P_NoHIns_enorm', 'P_POC_enorm']

# Calculate vulnerability index (average of normalized vulnerability indicators)
gdf['vulnerable'] = gdf[newvul].sum(axis='columns') / len(newvul)

# Calculate overall risk (hazard index * vulnerability index)
gdf['risk'] = gdf['vulnerable'] * gdf['hazard']


# Create a figure for plotting
fig = plt.figure(figsize=(16, 9))

# Define subplot layout (3 rows, 6 columns, specific width ratios)
gs = GridSpec(ncols=6,
              nrows=3,
              width_ratios=(20, 20, 20, 20, 20, 20),
              figure=fig)

# Define colormaps for each data category plot
cmaps = ['Blues', 'Reds', 'Greens']

# Define which plots should have legends
legends = (0, 1,)

# Loop through exposure, hazard, and vulnerability data categories
for m, indicator in enumerate((exposure, hazard, vulnerable)):

    print(m)
    for n, col in enumerate(indicator):
        ax = fig.add_subplot(gs[m, n])

        gdf.plot(column=col + 'norm',
                 cmap=cmaps[m],
                 ax=ax,
                 legend='brief')
        ax.axis('off')
        ax.set_title(labels[col], fontsize=8, loc='left')



gdf.to_file('risks_v2.shp')
fig.savefig('components.png', dpi=250, bbox_inches='tight')
