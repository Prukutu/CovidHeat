import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

import numpy as np
import geopandas as gpd
import scipy.io.netcdf as nc
import xarray as xr

import namelist


def draw_nest(ax, params, dom=2, linecolor='black', label=True):
    """
    Draw nested domains on a given axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis on which to draw the nested domains.
    - params (dict): Parameters from the WRF namelist.
    - dom (int): Domain level to draw (default is 2).
    - linecolor (str): Color of the lines (default is 'black').
    - label (bool): Whether to label the nested domain (default is True).
    """
    dom=dom-1
    extent = get_extent(params, dom=dom+1)

    dx_0 = float(params['dx'][0])
    dy_0 = float(params['dy'][0])
    ratios = [float(val) for val in params['parent_grid_ratio'][:-1]]

    dx = dx_0/np.prod(ratios[:dom+1])
    dy = dy_0/np.prod(ratios[:dom+1])

    xpoints = float(params['e_we'][dom])
    ypoints = float(params['e_sn'][dom])

    width = xpoints*dx
    height =  ypoints*dy

    ax.add_patch(mpatches.Rectangle(xy=(extent[0], extent[2]),
                                    width=width,
                                    height=height,
                                    facecolor='none',
                                    edgecolor=linecolor))

    if label:
        # write label above top left corner of box
        txt = 'D0' + str(dom+1)
        ax.text(extent[0],
                extent[2] + height + 10000,
                txt,
                fontsize=8,
                color=linecolor)


def get_extent(params, dom=1):
    """
    Get the extent of the nested domain.

    Parameters:
    - params (dict): Parameters from the WRF namelist.
    - dom (int): Domain level (default is 1).

    Returns:
    - tuple: Extent of the nested domain (lower_x, upper_x, lower_y, upper_y).
    """
    dom = dom - 1  # python counts on 0
    xpoints = float(params['e_we'][dom])
    ypoints = float(params['e_sn'][dom])

    dx_0 = float(params['dx'][0])
    dy_0 = float(params['dy'][0])
    ratios = [float(val) for val in params['parent_grid_ratio'][:-1]]

    dx = dx_0/np.prod(ratios[:dom+1])
    dy = dy_0/np.prod(ratios[:dom+1])

    # Divide domain 1 resolution by parent_grid_ratio

    lower_x = -dx*xpoints/2
    lower_y = -dy*ypoints/2

    upper_x = dx*xpoints/2
    upper_y = dy*ypoints/2

    return (lower_x, upper_x, lower_y, upper_y)


def USLColorMaps(color1, color2, n=5):
    """
    Create a colormap with USL colors.

    Parameters:
    - color1 (str): Start color.
    - color2 (str): End color.
    - n (int): Number of colors (default is 5).

    Returns:
    - matplotlib.colors.LinearSegmentedColormap: Colormap with USL colors.
    """
    cmap = colors.LinearSegmentedColormap.from_list('usl-colors',
                                                    [color1, color2],
                                                    N=n)
    return cmap


def loadNCfile(filename, varname):
    """
    Load data from a NetCDF file.

    Parameters:
    - filename (str): Path to the NetCDF file.
    - varname (str): Variable name to load.

    Returns:
    - numpy.ndarray: Loaded data.
    """
    ds = xr.load_dataset(filename)
    data = ds[varname].values.squeeze()
    return data


def getbounds(lower, upper, num=5):
    """
    Generate bounds for a given range.

    Parameters:
    - lower (float): Lower bound of the range.
    - upper (float): Upper bound of the range.
    - num (int): Number of bounds (default is 5).

    Returns:
    - numpy.ndarray: Bounds array.
    """
    return np.linspace(lower, upper, num=num)


def axlabel(ax, label, fs=8):
    """
    Add label to an axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to add the label.
    - label (str): Label text.
    - fs (int): Font size (default is 8).
    """
    xbounds = ax.get_xbound()
    ybounds = ax.get_ybound()

    xloc = xbounds[1] - (xbounds[1] - xbounds[0]) * .1
    yloc = ybounds[0] + (ybounds[1] - ybounds[0]) * .03

    ax.text(xloc, yloc, label, fontsize=fs)


# Load the MapPLUTO building info
varname = ['BUILD_AREA_FRACTION', 'BUILD_HEIGHT',
           'BUILD_SURF_RATIO', 'LU_INDEX',]
filename = ['bldgfrac.nc', 'bldghgt.nc', 'surfratio.nc', 'lu.nc']
data = {vname: loadNCfile(f, vname) for vname, f in zip(varname, filename)}
data['BUILD_AREA_FRACTION'] = data['BUILD_AREA_FRACTION']*100

lon, lat = [nc.netcdf_file('bldgfrac.nc').variables[v].data
            for v in ('XLONG', 'XLAT')]

# Boundary for the small axes
boundary = gpd.read_file('forecast-counties_v3.shp')
boros = gpd.read_file('nyc_5boroughs.shp').set_index('NAME')

# add centroids to boros shapefile
boros['centroid'] = boros.centroid

nl = namelist.Namelist(program='wps').load()

params = nl['&geogrid']
proj = ccrs.LambertConformal(central_longitude=float(params['ref_lon'][0]),
                             central_latitude=float(params['ref_lat'][0]),
                             standard_parallels=(float(params['truelat1'][0]),
                                                 float(params['truelat2'][0])))

fig = plt.figure(figsize=(7, 7))

# Initialize the grid spec
nrows = 4
ncols = 4
gs = gridspec.GridSpec(nrows, ncols,
                       figure=fig,
                       width_ratios=(8, 8, 10, 10))

# Domains axis
ax1 = fig.add_subplot(gs[:2, :2], projection=proj)

# borough map axis
ax4 = fig.add_subplot(gs[:2, 2:], projection=proj)

# input data axes
ax2 = fig.add_subplot(gs[2, 0], projection=proj)
ax3 = fig.add_subplot(gs[2, 1], projection=proj)
ax5 = fig.add_subplot(gs[2, 2], projection=proj)
ax6 = fig.add_subplot(gs[2, 3], projection=proj)

axes = (ax1, ax2, ax3, ax5)

# For ax1, we draw the total parent domain, then draw two circles,
# one for each nested domain
ax1.add_feature(feature.LAND.with_scale('10m'), facecolor='#BBBBBB')
ax1.set_extent(get_extent(params, dom=1), crs=proj)
draw_nest(ax1, params, dom=2, linecolor='#555456')
draw_nest(ax1, params, dom=3, linecolor='#555456')
ax1.grid('on')

# Plot the boros map
boros.plot(facecolor='#BBBBBB',
           edgecolor='#555456',
           linewidth=.5,
           ax=ax4)
ax4.axis('off')

# Add labels to each boro name using geometry centroids
boro_labels = {'Bronx': 'BX',
               'Kings': 'BK',
               'New York': 'MN',
               'Queens': 'QN',
               'Richmond': 'SI'}

for boro in boro_labels.keys():
    x, y = boros.loc[boro, 'centroid'].x, boros.loc[boro, 'centroid'].y
    ax4.text(x, y, boro_labels[boro],
             fontsize=8,
             family='IBM Plex Sans',
             ha='center')

# Define colormaps and bounds for each of the variables.
num = 8
bounds = {'BUILD_AREA_FRACTION': getbounds(0, 70, num=num),
          'BUILD_HEIGHT': getbounds(0, 120, num=num),
          'BUILD_SURF_RATIO': getbounds(0, 2.41, num=num)}

cmaps = {'BUILD_AREA_FRACTION': ('#CCE0EB', '#016699'),
         'BUILD_HEIGHT': ('#E4E9D3', '#769022'),
         'BUILD_SURF_RATIO': ('#FFEBCC', '#F99C1C')}

labelnames = {'BUILD_AREA_FRACTION': 'Building area (%)',
              'BUILD_HEIGHT': 'Building height (m)',
              'BUILD_SURF_RATIO': 'Surface Area Ratio',
              'LU_INDEX': 'Landuse index'}

extent = (-74.28344116210936, -73.66589965820313,
          40.47572402954101, 40.93720169067384)

fontsize = 8
pad = 3
varnames = ()
for vname, ax in zip(varname, axes[1:]):
    cmap = USLColorMaps(cmaps[vname][0], cmaps[vname][1], n=num)
    to_plot = data[vname]
    to_plot[to_plot == 0] = np.nan

    boundary.plot(edgecolor='black',
                  facecolor='none',
                  linewidth=0.33,
                  ax=ax,
                  zorder=1)
    im = ax.contourf(lon, lat, to_plot, bounds[vname],
                     cmap=cmap,
                     zorder=0,
                     extend='both')

    cbar = fig.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        drawedges=True)
    cbar.ax.tick_params(labelsize=6, rotation=45)
    ax.set_extent(extent, crs=proj)

    cbar.ax.set_title(labelnames[vname],
                      loc='left',
                      pad=pad,
                      fontsize=fontsize)


# LU_INDEX is gategorical so requires sloightly different approach.
landuse_colors = ('#E9DDD5', '#BC9880', '#90532C')
lu = data['LU_INDEX']
lu[lu < 31] = np.nan
lu[lu==31] = 0
lu[lu==32] = .5
lu[lu==33] = 1

cmap = colors.ListedColormap(landuse_colors)

im = ax6.pcolormesh(lon, lat, data['LU_INDEX'],
                    cmap=cmap)

cbar = fig.colorbar(im,
                    ax=ax6,
                    orientation='horizontal',
                    drawedges=True)
cbar.set_ticks([.175, 0.5, .825])
ax6.set_extent(extent, crs=proj)
cbar.ax.set_xticklabels(['Low Res.',
                         'High Res.',
                         'Comm.'],
                        rotation=45,
                        ha='right',
                        fontsize=fontsize)
boundary.plot(edgecolor='black',
              facecolor='none',
              linewidth=0.33,
              ax=ax6,
              zorder=1)
cbar.ax.set_title(labelnames['LU_INDEX'],
                  fontsize=fontsize,
                  loc='left',
                  pad=pad)

labels = ('a', 'b', 'c', 'd', 'e', 'f')
axes = (ax1, ax4, ax2, ax3, ax5, ax6)
for ax, label in zip(axes, labels):
    axlabel(ax, label)

fig.savefig('input_data_domain_wide.png', bbox_inches='tight')
fig.savefig('input_data_domain_hires_wide.png', bbox_inches='tight', dpi=250)

