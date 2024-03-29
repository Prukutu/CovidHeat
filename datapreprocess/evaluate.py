import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

import string


import geopandas as gpd  # Library for working with geospatial data
import pandas as pd  # Library for data analysis and manipulation
import xarray as xr  # Library for working with gridded data
import matplotlib.pyplot as plt  # Library for creating visualizations
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm  # For colormaps
from mpl_toolkits.basemap import CCRS  # For geographic projections
import string  # For subplot labels


def axlabel(ax, label, fs=12):
  """
  This function adds a label to a matplotlib axis at a specific location.

  Args:
      ax: The matplotlib axis to add the label to.
      label: The text label to be added.
      fs: The font size of the label (default 12).
  """
  xbounds = ax.get_xbound()
  ybounds = ax.get_ybound()

  xloc = xbounds[0] + (xbounds[1] - xbounds[0]) * 0.03
  yloc = ybounds[0] + (ybounds[1] - ybounds[0]) * 0.85

  ax.text(xloc, yloc, label, fontsize=fs)


def resample_asos(df, freq='H'):
  """
  This function resamples an ASOS station data DataFrame to a specified frequency.

  Args:
      df: The pandas DataFrame containing ASOS station data.
      freq: The desired resampling frequency (default 'H' for hourly).

  Returns:
      A new pandas DataFrame with the resampled data.
  """
  df = df.set_index('valid')
  df_resampled = df.resample(freq).mean()

  # Subset to only summer (JJA)
  return df_resampled.loc['2020-05-22':'2020-08-31 21:00']


def load_nc(filename, lon, lat, varname='T2'):
  """
  This function loads a NetCDF file, extracts a specific variable for a given 
  longitude and latitude, and converts it to a pandas DataFrame.

  Args:
      filename: The path to the NetCDF file.
      lon: The longitude of the desired location.
      lat: The latitude of the desired location.
      varname: The name of the variable to extract (default 'T2').

  Returns:
      A pandas DataFrame containing the extracted variable data.
  """
  ds = xr.load_dataset(filename)

  # Find closest grid point to airport
  distance = np.sqrt((ds.XLONG.values - lon) ** 2 +
                    (ds.XLAT.values - lat) ** 2)
  loc = np.where(distance == distance.min())
  ds['T2'] = (ds['T2'] - 273.15)  # Convert Kelvin to Celsius
  return ds[varname].isel(z=0, y=loc[0][0], x=loc[1][0]).to_dataframe()



nyc = gpd.read_file('nyc_geo.shp')
region = gpd.read_file('forecast-counties_v3.shp')
# nyc = nyc.to_crs(2263)

stations = pd.read_csv('asos_stations/asos.txt')
stations['valid'] = pd.to_datetime(stations['valid'])
station_IDs = stations['station'].unique()
lons = {ID: stations[stations['station'] == ID]['lon'].unique() for
        ID in station_IDs}
lats = {ID: stations[stations['station'] == ID]['lat'].unique() for
        ID in station_IDs}
# Extract T2 from wrfout
t2 = {ID: load_nc('wrfoutput/t2.d03.nc', lons[ID], lats[ID]) for
      ID in station_IDs}

# Load daily tmax data and compute summer average
tmax = xr.load_dataset('wrfoutput/t2.daymax.d03.nc')
tmax_mean = tmax['T2'].mean(axis=0) - 273.15

# Resample ASOS station data to hourly to match WRF output
asos_resampled = {ID: resample_asos(stations[stations['station'] == ID]) for
                  ID in station_IDs}


plt.style.use('../../../mpl_styles/usl-presentations')

fig = plt.figure(figsize=(7, 6))
proj = ccrs.epsg(2263)
# fig2, ax2 = plt.subplots(figsize=(5, 5))
bounds = np.arange(21, 32, 1.5)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('ClaudiaReds',
                                                           ['white',
                                                            # '#EB9954',
                                                            '#BE3526'],
                                                           N=len(bounds))
norm = matplotlib.colors.BoundaryNorm(bounds,
                                      ncolors=cmap.N,
                                      clip=False)

gs = GridSpec(ncols=2,
              nrows=4,
              width_ratios=(1.5, 1),
              figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2, ax3, ax4 = [fig.add_subplot(gs[n, 0], sharex=ax1) for n in range(1, 4)]

axes = (ax1, ax2, ax3, ax4)
mapax1 = fig.add_subplot(gs[0:2, 1])
mapax2 = fig.add_subplot(gs[2:, 1])
mapax1.axis('off')
mapax2.axis('off')
# Draw the NYC map
nyc.plot(facecolor='#D9D9D9',
         # edgecolor='#555456',
         ax=mapax1)

im = mapax2.contourf(tmax.XLONG, tmax.XLAT, tmax_mean.values, bounds,
                     cmap=cmap,
                    )
cbar = plt.colorbar(im, orientation='horizontal')
cbar.ax.set_title('2m Daily Max Temperature (\u00b0C)',
                  fontsize=10)
region.plot(ax=mapax2,
            facecolor='none',
            edgecolor='black',
            linewidth=.5)
mapax2.set_xlim(tmax.XLONG.min(), tmax.XLONG.max())
mapax2.set_ylim(tmax.XLAT.min(), tmax.XLAT.max())

# Colors for timeseries
colors = ['#016699', '#F99C1C', '#769022', '#47017F']
linecolors = ['#668FA3', '#FFC466', '#ADBC7A', '#9167B2']


for n, ID in enumerate(station_IDs):
    print(ID)
    axes[n].plot(t2[ID].index, t2[ID]['T2'],
                 color=colors[n],
                 linestyle='-',
                 linewidth=.5)
    axes[n].plot(asos_resampled[ID].index, asos_resampled[ID]['tmpc'],
                 color=linecolors[n],
                 linestyle='none',
                 marker='o',
                 markersize=.5)

    axes[n].set_ylim(5, 36)
    axes[n].set_ylabel(u'Temperature (\u00b0C)', fontsize=10)
    axes[n].set_title(ID, loc='left', fontsize=12)

    mapax1.scatter(lons[ID], lats[ID],
                   facecolor=colors[n],
                   edgecolor='none')


    # Compute MAE
    mae = np.abs(asos_resampled[ID]['tmpc'] - t2[ID]['T2']).mean()
    print(mae)
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.get_xticklabels(),
             visible=False)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

# Add labels to all the subplots
for n, ax in enumerate(fig.get_axes()[:6]):
    axlabel(ax, string.ascii_lowercase[n], fs=10)

fig.savefig('timeseries.png', bbox_inches='tight')
