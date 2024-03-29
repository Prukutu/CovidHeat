import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import libpysal as lps
import esda
import mapclassify as mc


def axlabel(ax, label, fs=12):
    xbounds = ax.get_xbound()
    ybounds = ax.get_ybound()

    xloc = xbounds[0] + (xbounds[1] - xbounds[0])*.8
    yloc = ybounds[0] + (ybounds[1] - ybounds[0])*.05

    ax.text(xloc, yloc, label, fontsize=fs)
    

np.random.seed(12345)
plt.style.use('usl-presentations')

df = gpd.read_file('risks_v2.shp')

# Remove values with NaN risk
heat = 't2norm'
covid = 'case_raten'
risk = 'risk'

# Nans belong to parks or uninhabited areas so convert to zero
df['risk'] = df['risk'].replace(np.nan, 0)

# Get spatial weights
wq = lps.weights.Queen.from_dataframe(df)
wq.transform = 'r'
# Remove islands from dataset and recompute weights
df = df.drop(wq.islands)
wq = lps.weights.Queen.from_dataframe(df)
wq.transform = 'r'

ax = df.plot(edgecolor='grey', facecolor='w')
f,ax = wq.plot(df, ax=ax, 
        edge_kws=dict(color='r', linestyle=':', linewidth=1),
        node_kws=dict(marker=''))
ax.set_axis('off')

plt.show()
y = df[risk]
# 
# # Plot the Moran Scatter plot
# lag_risk = lps.weights.lag_spatial(wq, df['risk'])
# 
# # Linear fit
# b, a = np.polyfit(df['risk'], lag_risk, 1)
# f, ax = plt.subplots(1, figsize=(4, 4))
# 
# plt.plot(df['risk'],
#          lag_risk,
#          marker='.',
#          linestyle='none',
#          color='#999899',
#          markersize=8,
#          alpha=0.5)
# 
#  # dashed vert at mean of the df['risk']
# plt.vlines(df['risk'].mean(),
#            lag_risk.min(),
#            lag_risk.max(),
#            linestyle='--',
#            color='#000000')
#  # dashed horizontal at mean of lagged price
# plt.hlines(lag_risk.mean(),
#            y.min(),
#            df['risk'].max(),
#            linestyle='--',
#            color='#000000')
# 
# # red line of best fit using global I as slope
# plt.plot(df['risk'],
#          a + b*df['risk'],
#          color='#336985')
# # ax.set_title('Moran Scatterplot', )
# ax.set_ylabel('Spatial lag of risk', fontsize=12)
# ax.set_xlabel('Risk', fontsize=12)
# f.savefig('moranscatter_v2.png', dpi=200)
# # Get the local Moran's I
# li = esda.moran.Moran_Local(y, wq, permutations=9999)
# mi = moran = esda.moran.Moran(y, wq, permutations=9999)
# 
# siglev = 0.05
# sig = 1 * (li.p_sim < siglev)
# hotspot = 1 * (sig * li.q==1)
# coldspot = 2 * (sig * li.q==3)
# # doughnut = 2 * (sig * li.q==2)
# # diamond = 4 * (sig * li.q==4)
# spots = hotspot + coldspot
# # spot_labels = [ 'No Significance', 'Hot Spot', 'Cold Spot', 'dough', 'diamond']
# 
# spot_labels = ['No significance', 'Hot spot', 'Cold spot']
# 
# labels = [spot_labels[i] for i in spots]
# 
# 
# df['cl'] = labels
# # df['colors'] = [colors[cat] for cat in df['cl'].values]
# df['spots'] = spots
# 
# hmap = colors.ListedColormap(['#016699', '#BE3526', '#DDDDDD'])
# 
# 
# f, ax = plt.subplots(1, figsize=(6, 6))
# 
# leg_kwarg = {'loc': 'upper left',
#              'fontsize': 10}
# df.plot(column='cl',
#         categorical=True,
#         k=3,
#         cmap=hmap,
#         # color=df['colors'],
#         linewidth=0.5,
#         ax=ax,
#         edgecolor='black',
#         legend=True,
#         legend_kwds=leg_kwarg
#         )
# 
# ax.set_title('Overlapping COVID-19 + Heat Risk',
#              fontsize=12,
#              loc='left')
# ax.set_axis_off()
# f.savefig('multihazard_risk_hotspot_v2.png', bbox_inches='tight')
# 
# 
# # 2-panel plot of the significance and the local Moran's I values
# f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 5))
# 
# hmap = colors.ListedColormap(['#DDDDDD', 'black'])
# 
# # Create a significance column in our dataframe and rename
# df['sig_p05'] = sig
# df['sig_p05'].replace(1, 'Significant (p < 0.05)', inplace=True)
# df['sig_p05'].replace(0, 'Not significant', inplace=True)
# 
# df['MI'] = li.Is
# 
# leg_kwarg = {'loc': 'upper left',
#              'fontsize': 10}
# 
# df.plot(column='sig_p05',
#         categorical=True,
#         k=2,
#         cmap=hmap,
#         # color=df['colors'],
#         linewidth=0.5,
#         ax=ax2,
#         edgecolor='#BBBBBB',
#         legend=True,
#         legend_kwds=leg_kwarg
#         )
# 
# ax1.axis('off')
# 
# leg_kwarg = {'loc': 'upper left',
#              'fontsize': 10,
#              'title': "Local Moran's I"}
# # Colormap for the Moran's I. Negative values have a different color
# hmap = colors.ListedColormap(['#FFD799',
#                               '#B599CC',
#                               '#9167B2',
#                               '#6C3499',
#                               '#47017F'])
# 
# df.plot(column='MI',
#         categorical=False,
#         k=5,
#         scheme='quantiles',
#         cmap=hmap,
#         # color=df['colors'],
#         linewidth=0.5,
#         ax=ax1,
#         edgecolor='black',
#         legend=True,
#         legend_kwds=leg_kwarg
#         )
# ax2.axis('off')
# 
# axlabel(ax2, 'b', fs=12)
# axlabel(ax1, 'a', fs=12)
# f.savefig('signif_localMoran_v2.png')
