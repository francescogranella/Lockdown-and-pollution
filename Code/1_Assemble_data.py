import glob
from pprint import pprint as print

import context
import numpy as np
import pandas as pd
from tqdm import tqdm

context.pdsettings()


def windvec(u=np.array([]), \
            D=np.array([])):
    '''
    Credits: http://python.hydrology-amsterdam.nl/modules/meteolib.py

    Function to calculate the wind vector from time series of wind
    speed and direction.

    Parameters:
        - u: array of wind speeds [m s-1].
        - D: array of wind directions [degrees from North].

    Returns:
        - uv: Vector wind speed [m s-1].
        - Dv: Vector wind direction [degrees from North].

    Examples
    --------

        >>> u = np.array([[ 3.],[7.5],[2.1]])
        >>> D = np.array([[340],[356],[2]])
        >>> windvec(u,D)
        (4.162354202836905, array([ 353.2118882]))
        >>> uv, Dv = windvec(u,D)
        >>> uv
        4.162354202836905
        >>> Dv
        array([ 353.2118882])

    '''
    ve = 0.0  # define east component of wind speed
    vn = 0.0  # define north component of wind speed
    D = D * np.pi / 180.0  # convert wind direction degrees to radians
    for i in range(0, len(u)):
        ve = ve + u[i] * np.sin(D[i])  # calculate sum east speed components
        vn = vn + u[i] * np.cos(D[i])  # calculate sum north speed components
    ve = - ve / len(u)  # determine average east speed component
    vn = - vn / len(u)  # determine average north speed component
    uv = np.sqrt(ve * ve + vn * vn)  # calculate wind speed vector magnitude
    # Calculate wind speed vector direction
    vdir = np.arctan2(ve, vn)
    vdir = vdir * 180.0 / np.pi  # Convert radians to degrees
    if vdir < 180:
        Dv = vdir + 180.0
    else:
        if vdir > 180.0:
            Dv = vdir - 180
        else:
            Dv = vdir
    return uv, Dv  # uv in m/s, Dv in dgerees from North


def egen_mean(df, var, by=None, where=None):
    if by:
        val = df.groupby(by)[var].transform(lambda x: np.mean(x))
    else:
        val = df[var].transform(lambda x: np.mean(x))
    if where is not None:
        val = np.where(where == True, val, np.nan)
    return val


def egen_total(df, var, by=None, where=None):
    if by:
        val = df.groupby(by)[var].transform(lambda x: np.sum(x))
    else:
        val = df[var].transform(lambda x: np.sum(x))
    if where is not None:
        val = np.where(where == True, val, np.nan)
    return val


##########################################################################################
# POLLUTION DATA

# Pollution data
def read_pollution():
    files = glob.glob(context.projectpath() + '/Data/Modified/Pollution20*.parq')
    dflist = [pd.read_parquet(x) for x in files]
    return pd.concat(dflist)


df = read_pollution()
df.rename(columns={'data': 'date'}, inplace=True)

# Drop invalid data
print(df['valid'].value_counts())
df = df.loc[df['valid'] == 1]
assert -9999 not in df['valore']

# Pollution metadata
meta = pd.read_parquet(context.projectpath() + '/Data/Modified/PollutionStations.parq')

meta = meta[pd.isna(meta['datastop']) == True]

# Merge pollution data and metadata
df = pd.merge(df, meta[['idsensore', 'idstazione', 'pollutant']], how='inner', on='idsensore', validate='m:1')

# Filter pollution data by pollutant
pollutants = ['Biossido di Azoto', 'PM10 (SM2005)', 'Particelle sospese PM2.5', 'PM10']
pollutants_dict = {'Biossido di Azoto': 'NO2', 'PM10 (SM2005)': 'PM10', 'Particelle sospese PM2.5': 'PM2.5',
                   'PM10': 'PM10'}

df = df.loc[df['pollutant'].isin(pollutants)]
df['pollutant'] = df['pollutant'].replace(pollutants_dict)

# Drop eventual duplicates
df.drop_duplicates(keep='first', inplace=True)

# Ratio PM 2.5 to PM 10
pm25 = df.loc[df['pollutant'] == 'PM2.5']
pm25 = pm25.groupby(['date', 'idstazione'])['valore'].mean().reset_index()
pm25.rename(columns={'valore': 'pm25'}, inplace=True)

pm10 = df.loc[df['pollutant'] == 'PM10']
pm10 = pm10.groupby(['date', 'idstazione'])['valore'].mean().reset_index()
pm10.rename(columns={'valore': 'pm10'}, inplace=True)

ratio = pd.merge(pm25, pm10, how='inner', on=['date', 'idstazione'], validate='1:1')

ratio['ratio'] = ratio['pm25'] / ratio['pm10']

ratio['dt'] = pd.to_datetime(ratio['date'].dt.date)
ratio = ratio.groupby(['dt', 'idstazione'])['ratio'].mean().reset_index()

monitors = ratio['idstazione'].unique()
renumber_dict = dict(zip(monitors, np.arange(1, len(monitors))))
ratio['idstazione'].replace(renumber_dict, inplace=True)
ratio['id'] = 'ratio_' + ratio['idstazione'].astype(str)
ratio = ratio.pivot(index='dt', columns='id', values='ratio').reset_index()

ratio.to_parquet(context.projectpath() + '/Data/Modified/Ratio.parq')

# Drop PM10
df = df.loc[df['pollutant'] != 'PM10']

# Daily avg
df['dt'] = pd.to_datetime(df['date'].dt.date)
df = df.groupby(['idsensore', 'pollutant', 'dt'])['valore'].mean().reset_index()

"""# Lags
df.sort_values(by=['idsensore', 'pollutant', 'dt'], inplace=True)

def lag(order=2):
    dflist = []
    for l in range(1, order+1):
        temp = df.groupby(['idsensore', 'pollutant'])['valore', 'dt'].shift(l)
        temp.columns = [f'valore_l{l}', f'dt_l{l}']
        temp.loc[df['dt'] != temp[f'dt_l{l}'] + pd.DateOffset(l), f'valore_l{l}'] = np.nan
        temp = temp[[f'valore_l{l}']]
        dflist.append(temp)
    return pd.concat(dflist, axis=1)

lags = lag(order=2)
df = pd.concat([df, lags], axis=1)"""

val_cols = [x for x in df.columns if 'valore' in x]
new_val_cols = [x.replace('valore', 'v') for x in val_cols]
rename_val_cols_dict = dict(zip(val_cols, new_val_cols))

df.rename(columns=rename_val_cols_dict, inplace=True)

# Reshape wide
df['id'] = df['idsensore'].astype(str) + '_' + df['pollutant']
df.head(3)

df = df.pivot(index='dt', columns='id', values=new_val_cols)
df.columns = [x[1] for x in df.columns]

df.reset_index(inplace=True)

# # Calendar variables
# df['year'] = df['dt'].dt.year
# df['month'] = df['dt'].dt.month
# df['day'] = df['dt'].dt.day
# df['week'] = df['dt'].dt.week
# df['dow'] = df['dt'].dt.dayofweek

# Drop columns with no data in 2020
df2020 = df.loc[df['dt'].dt.year == 2020]
no_2020_data_cols = df2020.sum()[df2020.sum() == 0].index.to_list()
print(f'No data in 2020 for {no_2020_data_cols}')
df.drop(columns=no_2020_data_cols, inplace=True)
del df2020

# Reorder columns
cols = df.columns.to_list()
new_cols = cols[:1] + cols[-5:] + cols[1:-5]
assert len(df.columns) == len(new_cols)
df = df[new_cols]

# Metadata
df.metadata = f"""Pollution data. Date 'dt' is unique identifier. Columns contain pollution values. Column names are <idsensore>_<pollutantshortname>
Data ranges from {df.dt.min()} to {df.dt.max()}"""

# Save
df.to_parquet(context.projectpath() + '/Data/Modified/Pollution.parq')

# end pollution data
##########################################################################################
#
# del df

##########################################################################################
# WEATHER DATA
# Weather data is much larger (longer) so it's processed year-by-year
files = glob.glob(context.projectpath() + '/Data/Modified/Meteo20*.parq')


def process(file):
    print('Reading ' + file[-13:])
    sendmsg('Reading ' + file[-13:])

    df = pd.read_parquet(file)

    print('File read. Now processing')

    # Drop invalid data
    # print(df['valid'].value_counts())
    df = df.loc[df['valid'] == 1]
    assert -9999 not in df['valore']

    df.drop(columns=['valid'], inplace=True)

    # Convert data to datetime format
    df.rename(columns={'data': 'date'}, inplace=True)
    if df.date.dtypes != '<M8[ns]':
        date_diff = (pd.to_datetime('1970-01-01') - pd.to_datetime(
            '1960-01-01'))  # Stata starts counting in 1960, pandas in 1970
        df['date'] = pd.to_datetime(df['date'] * 1000000) - date_diff

    # Weather metadata
    meta = pd.read_parquet(context.projectpath() + '/Data/Modified/MeteoStations.parq')

    # Merge weather data and metadata, automatically filtering by weather variable
    df = pd.merge(df, meta[['idsensore', 'idstazione', 'tipologia']], how='inner', on='idsensore', validate='m:1')

    # Outliers and invalid data
    mask1 = ((df['valore'] < 0) | (df['valore'] > 359)) & (df['tipologia'] == 'winddir')
    mask2 = ((df['valore'] < 0) | (df['valore'] > 20)) & (df['tipologia'] == 'prec')
    mask3 = ((df['valore'] < -20) | (df['valore'] > 40)) & (df['tipologia'] == 'temp')
    mask4 = ((df['valore'] < 0) | (df['valore'] > 100)) & (df['tipologia'] == 'hum')
    mask5 = ((df['valore'] < 0) | (df['valore'] > 50)) & (df['tipologia'] == 'windspeed')

    df = df[(~mask1) & (~mask2) & (~mask3) & (~mask4) & (~mask5)]

    # Measurement levels
    # """idOperatore: 	1 = Average value, 3 = Maximum value, 4 = Cumulative value, 2 = ?"""
    # print(pd.crosstab(df['idoperatore'], df['tipologia']))
    # print(df.groupby('tipologia')['valore'].describe())
    # Wind direction
    mask1 = (df['idoperatore'] != 1) & (df['tipologia'] == 'winddir')
    # Precipitation
    mask2 = (df['idoperatore'] != 4) & (df['tipologia'] == 'prec')
    # Temperature
    mask3 = (df['idoperatore'] != 1) & (df['tipologia'] == 'temp')
    # Humidity
    mask4 = (df['idoperatore'] != 1) & (df['tipologia'] == 'hum')
    # Widn speed
    mask5 = (df['idoperatore'] != 1) & (df['tipologia'] == 'windspeed')

    df = df[(~mask1) & (~mask2) & (~mask3) & (~mask4) & (~mask5)]

    df.drop(columns='idoperatore', inplace=True)

    # Daily average, sum, min, max
    df['dt'] = pd.to_datetime(df['date'].dt.date)

    # Average wind direction and speed
    # Pass speed to column for wind direction
    print('Average wind direction and speed')

    windspeed = df[df['tipologia'] == 'windspeed']
    wind = pd.merge(df, windspeed[['idstazione', 'date', 'valore']], how='inner', on=['idstazione', 'date'],
                    suffixes=('', 'speed'))

    wind = wind[wind['tipologia'] == 'winddir']
    grouped = wind.groupby(['idsensore', 'dt'])
    _ = grouped.apply(lambda x: windvec(x.valorespeed.values, x.valore.values))
    wind = pd.DataFrame(_.to_list(), index=_.index, columns=['windspeed', 'winddir'])

    windspeed = wind['windspeed'].reset_index()
    windspeed['stat'] = 'mean'
    windspeed['tipo'] = 'windspeed'
    windspeed.rename(columns={'windspeed': 'valore'}, inplace=True)

    winddir = wind['winddir'].reset_index()
    winddir['stat'] = 'mean'
    winddir['tipo'] = 'winddir'
    winddir.rename(columns={'winddir': 'valore'}, inplace=True)

    # Average, min and max temperature
    print('Average, min and max temperature')

    temp = df[df['tipologia'] == 'temp']
    temp = temp.groupby(['idsensore', 'dt']).agg(temp_mean=('valore', 'mean'), temp_min=('valore', 'min'),
                                                 temp_max=('valore', 'max')).reset_index()
    temp = pd.wide_to_long(temp, stubnames='temp', i=['idsensore', 'dt'], j='stat', sep='_', suffix='\w+').reset_index()
    temp['tipo'] = 'temp'
    temp.rename(columns={'temp': 'valore'}, inplace=True)

    # Humidity
    print('Average humidity')

    hum = df[df['tipologia'] == 'hum']
    hum = hum.groupby(['idsensore', 'dt']).agg(hum=('valore', 'mean')).reset_index()
    hum['stat'] = 'mean'
    hum['tipo'] = 'hum'
    hum.rename(columns={'hum': 'valore'}, inplace=True)

    # Precipitation
    print('Precipitation')

    prec = df[df['tipologia'] == 'prec']
    prec = prec.groupby(['idsensore', 'dt']).agg(prec=('valore', 'sum')).reset_index()
    prec['stat'] = 'sum'
    prec['tipo'] = 'prec'
    prec.rename(columns={'prec': 'valore'}, inplace=True)

    # Append back
    df = pd.concat([windspeed, winddir, temp, hum, prec])
    return df


dflist = []
for i, file in enumerate(tqdm(files)):
    fileyear = file[-9:-5]
    import os

    if os.path.isfile(context.projectpath() + f'/Data/{fileyear}.parq'):
        temp = pd.read_parquet(context.projectpath() + f'/Data/{fileyear}.parq')
    else:
        temp = process(file)
        temp.to_parquet(context.projectpath() + f'/Data/{fileyear}.parq')
    dflist.append(temp)

# Append files of different years
df = pd.concat(dflist, axis=0)


# Lags
def lag(df, by, order=7):
    # Make sure the df is properly sorted.
    # Otherwise lags will be incorrect
    if not isinstance(by, list):
        by = [by]
    df.sort_values(by=by + ['dt'], inplace=True)
    dflist = []
    # For every order of lags, find lagged values
    for l in range(1, order + 1):
        # By 'idsensore', 'tipo', 'stat': shift value and date
        temp = df.groupby(by)['valore', 'dt'].shift(l)
        # Rename columns to differ from columns of original values
        temp.columns = [f'valore_l{l}', f'dt_l{l}']
        # Fill with missing where difference in days b/w original and lagged date is incorrect.
        # This happens when data has gaps
        temp.loc[df['dt'] != temp[f'dt_l{l}'] + pd.DateOffset(l), f'valore_l{l}'] = np.nan
        # Now we only care about the lagged values
        temp = temp[[f'valore_l{l}']]
        # Append to list
        dflist.append(temp)
    # Concatenate horizontally
    lags = pd.concat(dflist, axis=1)
    return pd.concat([df, lags], axis=1)


df = lag(df, by=['idsensore', 'tipo', 'stat'])

val_cols = [x for x in df.columns if 'valore' in x]
new_val_cols = [x.replace('valore', '') for x in val_cols]
rename_val_cols_dict = dict(zip(val_cols, new_val_cols))

# Note that column of 'current' values has no name.
#  This is intentional
df.rename(columns=rename_val_cols_dict, inplace=True)
print(df.head(3))

# Reshape wide
df['id'] = df['idsensore'].astype(int).astype(str) + '_' + df['tipo'] + '_' + df['stat']
df = df.pivot(index='dt', columns='id', values=new_val_cols)
df.columns = [x[1] + x[0] for x in df.columns]

df.reset_index(inplace=True)

print(df[df.columns[:10]].head(3))

# Metadata
df.metadata = f"""Weather data. Date 'dt' is unique identifier. Columns contain weather values. Column names are <idsensore>_<weathervar>_<aggregationstat>
Data ranges from {df.dt.min()} to {df.dt.max()}"""

# Save
df.to_parquet(context.projectpath() + '/Data/Modified/Weather.parq')

# end weather data
##########################################################################################

del df

##########################################################################################
# ATMOSPHERIC SOUNDING
df = pd.read_stata(context.projectpath() + '/Data/Modified/AtmoSounding.dta')

# Subset
subset_cols = ['Showalter_index', 'Lifted_index', 'SWEAT_index', 'K_index', 'Cross_totals_index',
               'Vertical_totals_index']  # , 'Totals_totals_index', ]
df = df[['date'] + subset_cols]

# Drop duplicates
df.drop_duplicates(inplace=True)

# Date, hour
df['dt'] = df['date'].dt.date
df['hour'] = df['date'].dt.hour
# Only midnight and midday
df = df[df['hour'].isin([0, 12])]
df.drop(columns='date', inplace=True)

# Reshape wide
indices_cols = [x for x in df.columns if x != 'dt' and x != 'hour']
df = df.pivot(index='dt', columns='hour', values=indices_cols)
df.columns = [x[0] + str(x[1]) for x in df.columns]

print(df[df.columns[:5]].head(3))

# Reindex: fill gaps in the time series with missing values
rindex = pd.date_range(df.index.min(), df.index.max())
df = df.reindex(rindex)

# Lags
# Make sure ordered by date
df.sort_index(inplace=True)
dflist = []
for i in range(1, 8):
    temp = df.shift(i)
    temp.columns = [x + f'_l{i}' for x in temp.columns]
    dflist.append(temp)

df = pd.concat([df] + dflist, axis=1)

df.index.name = 'dt'
df.reset_index(inplace=True)

# Save
df.to_parquet(context.projectpath() + '/Data/Modified/AtmoSoundingIndices.parq')

# end atmospheric sounding data
##########################################################################################

del df

##########################################################################################
# MERGE POLLUTION, WEATHER, ATMOSPHERIC SOUNDING

pollution = pd.read_parquet(context.projectpath() + '/Data/Modified/Pollution.parq')
meteo = pd.read_parquet(context.projectpath() + '/Data/Modified/Weather.parq')
atmo = pd.read_parquet(context.projectpath() + '/Data/Modified/AtmoSoundingIndices.parq')
ratio = pd.read_parquet(context.projectpath() + '/Data/Modified/Ratio.parq')

df = pd.merge(pollution, meteo, how='inner', on='dt', validate='1:1')
df = pd.merge(df, atmo, how='inner', on='dt', validate='1:1')
df = pd.merge(df, ratio, how='inner', on='dt', validate='1:1')

# Winsorize
data_cols = [x for x in df.columns if 'dt' not in x]
bottom, top = df[data_cols].quantile(0.01), df[data_cols].quantile(0.99)
df[data_cols] = df[data_cols].clip(bottom, top, axis=1)

"""# Holidays
carnival = ['2-23-2020', '3-3-2019', '2-11-2018', '2-26-2017', '2-7-2016', '2-15-2015', '3-2-2014', '2-10-2013']
carnival = [pd.to_datetime(x) for x in carnival]

easter = ['4-12-2020', '4-21-2019', '4-1-2018', '4-16-2017', '3-27-2016', '4-5-2015', '4-20-2014', '3-31-2013']
easter = [pd.to_datetime(x) for x in easter]

df['carnival'] = df['dt'].isin(carnival) * 1
df['easter'] = df['dt'].isin(easter) * 1

# Lagged holidays
df.sort_values('dt', inplace=True)
dflist = []
for i in range(1,8):
    temp = df[['carnival', 'easter']].shift(i)
    temp.columns = [x + f'_l{i}' for x in temp.columns]
    dflist.append(temp)

df = pd.concat([df] + dflist, axis=1)"""

# # Calendar variables
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['day'] = df['dt'].dt.day
df['week'] = df['dt'].dt.week
df['dow'] = df['dt'].dt.dayofweek
df['doy'] = df['dt'].dt.dayofyear
dow = pd.get_dummies(df['dt'].dt.dayofweek, prefix='dow', prefix_sep='')
week = pd.get_dummies(df['dt'].dt.week, prefix='week', prefix_sep='')
df = pd.concat([df, dow, week], axis=1)

# Seasonality
deg = df['doy'] / 366 * 360
for i in range(1, 13):
    print(i)
    df[f'sin{i}'] = np.sin(deg * np.pi / 180 + np.pi / 12 * i)

"""# Post April 15
df['heatingon'] = ((df['doy']<106) | (df['doy']>288))*1"""

print(df.shape)

# Save
df.to_parquet(context.projectpath() + '/Data/Modified/Pollution+Meteo.parq')

##########################################################################################
# end
