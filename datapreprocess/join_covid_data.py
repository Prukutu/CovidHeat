import geopandas as gpd
import pandas as pd
import xarray as xr

from shapely.geometry import Point

import matplotlib.pyplot as plt

def toGDF(lon, lat):
    geom = [Point(x, y) for x, y in zip(lon.flatten(), lat.flatten())]

    df = gpd.GeoDataFrame({'geometry': geom})

    df.crs = {'init': 'epsg:4326'}

    return df


df = pd.read_csv('data-by-modzcta_v2.csv')
sovi = pd.read_csv('covid_ZCTA_2018_SOVI_080520.csv')
zcta = gpd.read_file('MODZCTA_2010.shp')
zcta['MODZCTA'] = zcta['MODZCTA'].astype('int64')

ds = xr.load_dataset('wrfoutput/t2.daymax.d03.nc')

joined = zcta.merge(df, left_on='MODZCTA',
                    right_on='MODIFIED_ZCTA')
joined = joined.merge(sovi, left_on='MODZCTA',
                      right_on='GEOID10')
gdf = toGDF(ds.XLONG.values,
            ds.XLAT.values)
gdf['T2'] = ds['T2'].mean(axis=0).values.ravel()

gdf = gdf.to_crs(joined.crs)
gdf['geometry'] = gdf.buffer(750)
# Aggregate T2 values to the MODZCTAS
joined_t2 = gpd.sjoin(joined, gdf).groupby('MODZCTA').mean()

joined.set_index('MODZCTA', inplace=True)
joined['t2'] = joined_t2['T2']

# Subset the attribute table to avoid shapefile column label character limit
subset = joined[joined.columns[2:]]
subset.to_csv('nyc_heat_data_v2.csv')
