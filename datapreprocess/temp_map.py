import xarray as xr
import regionmask
import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

fname = 'wrfoutput/t2.daymax.d03.nc'
boundary = gpd.read_file('nyc_geo.shp')
proj = ccrs.PlateCarree()

plt.style.use('../../../mpl_styles/usl-presentations')
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection=proj)


ds = xr.load_dataset(fname)

# Create mask for NYC data
mask = regionmask.mask_geopandas(boundary, ds['XLONG'], ds['XLAT'])

tmax = ds['T2'].mean(axis=0).where(mask==0) - 273.15
levels = np.arange(26, 33, .5)
im = ax.contourf(ds['XLONG'].values, ds['XLAT'].values, tmax.values, levels,
                 cmap='YlOrRd')
plt.colorbar(im)
boundary.plot(edgecolor='black', facecolor='none', ax=ax)

ax.set_title(u'Daily Maximum Temperature (\u00B0C)\nSummer (JJA) 2020',
             fontsize=16,
             loc='left')
ax.axis('off')

fig.savefig('daymax.png', bbox_inches='tight')
