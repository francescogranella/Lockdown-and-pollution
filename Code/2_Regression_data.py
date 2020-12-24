import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

import context
from idw import tree

context.pdsettings()

########################################################################################################################
df = pd.read_parquet(context.projectpath() + '/Data/Modified/Pollution+Meteo.parq')


# Metadata: coordinates from degrees to meters

def lnglat2meters(lng, lat):
    gdf = gpd.GeoSeries(gpd.points_from_xy(lng, lat), crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)
    xy = gdf.geometry.transform(lambda x: x.coords[0])
    return pd.DataFrame(xy.to_list(), columns=['x', 'y'])


########################################################################################################################
# POLLUTION

# Metadata
pollution_metadata = pd.read_parquet(context.projectpath() + '/Data/Modified/PollutionStations.parq')
# Only stations that work in 2020
pollution_metadata = pollution_metadata[pd.isna(pollution_metadata['datastop']) == True]
# Convert degrees to meters
xy = lnglat2meters(pollution_metadata['lng'], pollution_metadata['lat'])
pollution_metadata = pd.concat([pollution_metadata.reset_index(drop=True), xy.reset_index(drop=True)], axis=1)

# Pollution data proper
# NO2
no2_cols = [x for x in df.columns if 'NO2' in x]
revert_cols = ['_'.join(x.split('_')[1:]) + '_' + x.split('_')[0] for x in
               no2_cols]
no2 = df[['dt'] + no2_cols]
no2.columns = ['dt'] + revert_cols
no2 = pd.wide_to_long(no2, 'NO2', i='dt', j='idsensore', sep='_').reset_index()
no2.rename(columns={'NO2': 'val'}, inplace=True)

# PM2.5
pm25_cols = [x for x in df.columns if 'PM2.5' in x]
revert_cols = ['_'.join(x.split('_')[1:]) + '_' + x.split('_')[0] for x in
               pm25_cols]
pm25 = df[['dt'] + pm25_cols]
pm25.columns = ['dt'] + revert_cols
pm25 = pd.wide_to_long(pm25, 'PM2.5', i='dt', j='idsensore', sep='_').reset_index()
pm25.rename(columns={'PM2.5': 'val'}, inplace=True)

# NO2 and PM2.5
pollution = pd.concat([no2, pm25], axis=0)

# Add metadata
pollution = pd.merge(pollution, pollution_metadata, how='inner', on='idsensore', validate='m:1')

########################################################################################################################
# WEATHER

# Metadata
weather_metadata = pd.read_parquet(context.projectpath() + '/Data/Modified/MeteoStations.parq')
xy = lnglat2meters(weather_metadata['lng'], weather_metadata['lat'])
weather_metadata = pd.concat([weather_metadata.reset_index(drop=True), xy.reset_index(drop=True)], axis=1)

# Weather data proper
pollution_cols = [x for x in df.columns if 'NO2' in x or 'PM10' in x or 'PM2.5' in x]
weather_cols = [x for x in df.columns if
                'temp_' in x or 'prec_' in x or 'hum_' in x or 'winddir_' in x or 'windspeed_' in x if
                '_l' not in x]  # No lagged vars
revert_cols = ['_'.join(x.split('_')[1:]) + '_' + x.split('_')[0] for x in weather_cols]

atmo_cols = [x for x in df.columns if
             'Showalter_index' in x or 'Lifted_index' in x or 'LIFT_computed_using_virtual_temp' in x
             or 'SWEAT_index' in x or 'K_index' in x or 'Cross_totals_index' in x or 'Vertical_totals_index' in x
             or 'Totals_totals_index' in x]
sine_cols = [f'sin{x}' for x in range(1, 8)]
holiday_cols = [x for x in df.columns if 'carnival' in x or 'easter' in x]

weather = df[['dt'] + weather_cols]
weather.columns = ['dt'] + revert_cols

print(weather[weather.columns[:5]].head())

# Reshape long
weather.columns = [x.replace('.0', '') for x in weather.columns]


def stub_wide_to_long(df, stub):
    df = df[['dt'] + [x for x in df.columns if stub in x]]
    df = pd.wide_to_long(df, stub, i='dt', j='idsensore', sep='_').reset_index()
    df.rename(columns={stub: 'val'}, inplace=True)
    df['stat'] = stub
    return df


dflist = []
for stub in ['prec_sum', 'temp_max', 'temp_min', 'temp_mean', 'winddir_mean', 'windspeed_mean']:
    dflist.append(stub_wide_to_long(weather, stub))
weather = pd.concat(dflist)

# Add metadata
weather = pd.merge(weather, weather_metadata, how='inner', on='idsensore', validate='m:1')


########################################################################################################################
# FIND WEATHER STATION WITHIN RADIUS, COMPUTE IDW, BY DAY, BY TYPE


def idw_knn_withinradius(X1, X2, lon='lng', lat='lat', val='val', k=3, d=0.2):
    # # Generate data
    # points_with_values = pd.DataFrame(data=np.random.rand(100,3), columns=['lon', 'lat', 'val'])
    # points_without_values = pd.DataFrame(np.random.rand(2,2), columns=['lon', 'lat'])

    # Keep K nearest neighbour and/or neighbors within D degrees
    dist_tree = cKDTree(X1[[lon, lat]].values)
    # KNN
    dist, idx = dist_tree.query(X2[[lon, lat]].values, k=k)
    # Within radius
    X1 = X1.iloc[idx[dist < d]]

    # IDW
    if isinstance(val, list):
        vallist = []
        for v in val:
            idw = tree(X1[[lon, lat]].values, X1[[v]].values)
            vals = idw.transform(X2[[lon, lat]].values, k=k)
            vallist.append(vals)
        return vallist
    else:
        idw = tree(X1[[lon, lat]].values, X1[[val]].values)
        vals = idw.transform(X2[[lon, lat]].values, k=k)
        return vals


dflist = []
for tipo in weather['stat'].unique():
    ppchunks = []
    for i, day in enumerate(tqdm(pollution['dt'].unique())):
        pchunk = pollution[pollution['dt'] == day]
        mchunk = weather[(weather['dt'] == day) & (weather['stat'] == tipo)]
        vals = idw_knn_withinradius(mchunk, pchunk, lon='x', lat='y', val='val', k=3, d=25000)
        pchunk = pchunk[['dt', 'idsensore']]
        pchunk[f'{tipo}'] = vals
        pchunk.set_index(['dt', 'idsensore'], inplace=True)
        ppchunks.append(pchunk)
    tipochunk = pd.concat(ppchunks, axis=0)
    dflist.append(tipochunk)

df = pd.concat(dflist, axis=1).reset_index()

_len_df = len(df)

df = pd.merge(df, pollution, how='inner', on=['dt', 'idsensore'], validate='1:1')
df = pd.merge(df, pollution_metadata[['idsensore', 'pollutantshort']], how='inner', on='idsensore', validate='m:1')
assert len(df) == _len_df

df = df[['dt', 'idsensore', 'prec_sum', 'temp_max', 'temp_min', 'temp_mean', 'winddir_mean', 'windspeed_mean', 'val',
         'pollutant', 'nomestazione', 'datastart', 'datastop']]

df.to_parquet(context.projectpath() + '/Data/Modified/RegressionData.parq')
df.to_stata(context.projectpath() + '/Data/Modified/RegressionData.dta', write_index=False)
