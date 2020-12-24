import context
import geopandas as gpd
import geoplot as gplt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from shapely.geometry import Point
from utils import secondary_pm

context.pdsettings()

meteometafile = context.projectpath() + '/Data/Modified/MeteoStations.dta'
meteometadf = pd.read_stata(meteometafile).reset_index(drop=True)
pollmetafile = context.projectpath() + '/Data/Modified/PollutionStations.dta'
pollmetadf = pd.read_stata(pollmetafile)
meteofile = context.projectpath() + '/Data/Modified/MeteoLong.dta'
meteodf = pd.read_stata(meteofile)
pollfile = context.projectpath() + '/Data/Modified/PollutionLong.dta'
polldf = pd.read_stata(pollfile)

pollmetadf = pollmetadf[pd.isna(pollmetadf['stop'])]
pollmetadf = pollmetadf[pollmetadf['start'].dt.year < 2016]
pollmetadf = pollmetadf.loc[pollmetadf['pollutantshort'].isin(['PM2.5', 'NO2', 'BlackCarbon'])]
meteometadf = meteometadf[pd.isna(meteometadf['stop'])]
meteometadf = meteometadf[meteometadf['start'].dt.year < 2016]

lombardia = gpd.read_file(context.projectpath() + "/Data/Original/Shapefiles/ITA_adm1.shp")
milano = gpd.GeoSeries(Point(9.190406, 45.464454), crs='epsg:4326')

# Weather & pollution
geom = [Point(lon, lat) for lon, lat in zip(pollmetadf['lng'], pollmetadf['lat'])]
gdf = gpd.GeoDataFrame(pollmetadf, geometry=geom, crs='EPSG:4326')

geom = [Point(lon, lat) for lon, lat in zip(meteometadf['lng'], meteometadf['lat'])]
mgdf = gpd.GeoDataFrame(meteometadf, geometry=geom, crs='EPSG:4326')

# Europe and Lombardia
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
lombardia = gpd.read_file(context.projectpath() + "/Data/Original/Shapefiles/ITA_adm1.shp")
lomb = lombardia[lombardia['NAME_1'] == 'Lombardia']

# Plot Lombardia in Europe
fig, ax = plt.subplots(figsize=(12, 12))
gplt.polyplot(world, ax=ax, edgecolor='white', facecolor='lightgray', extent=lombardia.total_bounds + [-16, -1, 7, 20])
lomb.plot(ax=ax, facecolor='black', edgecolor='black')
plt.tight_layout()
plt.savefig(context.projectpath() + '/Docs/img/lomb_in_europe.png', dpi=100)
plt.show()

# Pop grid in lombardia
popgrid = gpd.read_file(
    context.projectpath() + r"/Data/Original/Shapefiles/GEOSTAT_grid_POP_1K_IT_2011-22-10-2018/GEOSTAT_grid_POP_1K_IT_2011.shp",
    mask=lomb)
popgrid = popgrid.to_crs(lomb.crs)
popgrid['lpop'] = np.log(popgrid['Pop'] + 1)

# Plot weather stations and pollution monitors over a pop grid

fig, ax = plt.subplots(figsize=(12, 12))
# Plot pop grid
popgrid.plot(ax=ax, column='Pop', cmap='Greys', legend=True, norm=LogNorm(vmin=1, vmax=popgrid['Pop'].max()),
             legend_kwds={'label': 'Population density (inhab./km$^2$)', 'orientation': 'horizontal', 'pad': 0.01,
                          'shrink': 0.5})
# Borders of Lombardia:
lomb.boundary.plot(ax=ax, edgecolor='black')
# Weather stations
mgdf.plot(ax=ax, alpha=1, markersize=45, edgecolor='k', linewidth=0.2, marker='X', color='red', label='Weather station',
          zorder=1)
# Pollution monitors
gdf.plot(ax=ax, alpha=1, markersize=45, edgecolor='k', linewidth=0.2, marker='o', color='blue',
         label='Pollution monitor', zorder=1)

# Add picture inside the plot
arr_img = plt.imread(context.projectpath() + '/Docs/img/lomb_in_europe.png', format='png')
# Locate
xy = [11, 46]
imagebox = OffsetImage(arr_img, zoom=0.15)
imagebox.image.axes = ax
ab = AnnotationBbox(imagebox, xy, xycoords='data', boxcoords="offset points", pad=0.5)
ax.add_artist(ab)
# Extend y axis to fit colorbar
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.2)
# Final touch
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
# plt.tight_layout(pad=0)
# Legend
import matplotlib.patches as mpatches

handles, labels = ax.get_legend_handles_labels()
patch = mpatches.Patch(color='silver', label='Population grid')
handles.append(patch)
plt.legend(handles=handles, loc='upper left', prop={'size': 15})
plt.savefig(context.projectpath() + '/Docs/img/pollution_meteo_stations.eps', bbox_inches='tight', dpi=200)
plt.show()

# Average pollution in Milan
whodict = {'PM2.5': 10, 'NO2': 40}
eudict = {'PM2.5': 25, 'NO2': 40}
df = polldf.merge(pollmetadf, how='inner', on='idsensore')
for poll in ['PM2.5', 'NO2']:
    ps = df.loc[df['pollutantshort'] == poll]
    ps = ps.loc[ps['comune'] == 'Milano']
    ps['doy'] = ps['dt'].dt.dayofyear

    historic = ps.loc[(ps['dt'].dt.year > 2014) & (ps['dt'].dt.year < 2020) & ((ps['doy'] <= 90) | (ps['doy'] >= 270))]
    historic = historic.groupby('doy')['pollution'].agg(min='min', max='max', mean='mean').reset_index()
    historic['dt'] = pd.to_datetime(historic['doy'] + 2020 * 1000, format='%Y%j')
    historic['dt'] = np.where(historic['doy'] <= 90, pd.to_datetime(historic['doy'] + 2020 * 1000, format='%Y%j'),
                              pd.to_datetime(historic['doy'] + 2019 * 1000, format='%Y%j'))
    historic.sort_values(by='dt', inplace=True)

    weeks = mdates.WeekdayLocator()
    weeks_fmt = mdates.DateFormatter('%W')
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')
    days = mdates.DayLocator()
    days_fmt = mdates.DateFormatter('%d %b')

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(historic['dt'], historic['mean'], alpha=1, linestyle='dashdot', label='5-year mean', zorder=2)
    ax.plot(historic['dt'], historic['min'], alpha=0.5, linestyle='dotted', color='black', label='5-year minmum',
            zorder=1)
    ax.plot(historic['dt'], historic['max'], alpha=0.35, linestyle='dashed', color='black', label='5-year maximum',
            zorder=1)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(days_fmt)
    ax.xaxis.set_major_formatter(days_fmt)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    ax.axhline(whodict[poll], color='black', zorder=1)
    ax.text(ax.get_xlim()[0] + 0.5, whodict[poll] + 2, 'WHO safety level')
    plt.ylabel('$\mu g/m^3$')
    plt.legend()
    plt.title('Winterime concentrations in Milan - ' + poll)
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/Average_{poll}.png')
    plt.show()

# Average pollution in Lombardia
whodict = {'PM2.5': 10, 'NO2': 40}
eudict = {'PM2.5': 25, 'NO2': 40}
df = polldf.merge(pollmetadf, how='inner', on='idsensore')
for poll in ['PM2.5', 'NO2']:
    ps = df.loc[df['pollutantshort'] == poll]
    ps['doy'] = ps['dt'].dt.dayofyear

    historic = ps.loc[(ps['dt'].dt.year > 2014) & (ps['dt'].dt.year < 2020) & ((ps['doy'] <= 90) | (ps['doy'] >= 270))]
    historic = historic.groupby(['idsensore', 'doy'])['pollution'].agg(min='min', max='max', mean='mean').reset_index()
    historic['dt'] = pd.to_datetime(historic['doy'] + 2020 * 1000, format='%Y%j')
    historic['dt'] = np.where(historic['doy'] <= 90, pd.to_datetime(historic['doy'] + 2020 * 1000, format='%Y%j'),
                              pd.to_datetime(historic['doy'] + 2019 * 1000, format='%Y%j'))
    historic.sort_values(by='dt', inplace=True)

    weeks = mdates.WeekdayLocator()
    weeks_fmt = mdates.DateFormatter('%W')
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')
    days = mdates.DayLocator()
    days_fmt = mdates.DateFormatter('%d %b')

    fig, ax = plt.subplots(figsize=(14, 4))
    for id in historic['idsensore'].unique():
        temp = historic.loc[historic['idsensore'] == id]
        ax.plot(temp['dt'], temp['mean'], alpha=0.10, color='black', zorder=1)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(days_fmt)
    ax.xaxis.set_major_formatter(days_fmt)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    ax.axhline(whodict[poll], color='black', zorder=1)
    ax.text(ax.get_xlim()[0] + 0.5, whodict[poll] - 5, 'WHO safety level')
    plt.ylabel('$\mu g/m^3$')
    plt.title('Winterime concentrations in Lombardia - ' + poll + '\nEach line is a monitoring station')
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/Average_Lombardia_{poll}.png')
    plt.show()

# Date formatters
weeks = mdates.WeekdayLocator()
weeks_fmt = mdates.DateFormatter('%W')
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b')
days = mdates.DayLocator()
days_fmt = mdates.DateFormatter('%d %b')


def format_ticks(ax):
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(days_fmt)


def rotate_ticks(fontsize=12):
    plt.xticks(rotation=90, ha='center', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)


# Relevant moments
codogno = pd.to_datetime('2020-02-21')
schools_closed = pd.to_datetime('2020-02-24')
lockdown = pd.to_datetime('2020-03-08')
dust_start = pd.to_datetime('2020-03-28')
dust_end = pd.to_datetime('2020-03-29')


def vertical_lines(ax):
    ax.axvline(codogno, color='black')
    ax.axvline(schools_closed, color='black')
    ax.axvline(lockdown, color='black')


def place_text(ax, offset=-18, fontsize=12):
    ylims = ax.get_ylim()
    try:
        if poll == 'PM2.5':
            ylims = (0, 80)
    except:
        pass
    ax.set_ylim(ylims)
    max = ylims[1]
    fontsize = 12
    ax.text(codogno + pd.DateOffset(offset), max * 0.85, "           Lockdown\n11 municipalities", fontsize=fontsize)
    ax.text(schools_closed, max * 0.85, " Schools\n close", fontsize=fontsize)
    ax.text(lockdown, max * 0.85, " Lockdown\n Lombardy", fontsize=fontsize)


# Secondary PM
# 2017-2019
df = pd.read_excel(
    r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\Covid\Data\Original\Pollution\MI_composizione_2017_2019.xlsx",
    sheet_name='MI_Pascal PM10_IC', header=7)
df = df[list(df.columns[3:4]) + ['PM', 'NO3-', 'SO42-', 'NH4+']]
df.rename(columns={df.columns[0]: 'date'}, inplace=True)
df = df.drop(0)
df['date'] = pd.to_datetime(df['date'])
for col in ['PM', 'NO3-', 'SO42-', 'NH4+']:
    df.loc[df[col].astype(str).str.contains('---'), col] = np.nan
    df.loc[df[col].astype(str).str.contains('<'), col] = 0
    df[col] = pd.to_numeric(df[col])
df = df.loc[df['date'] > pd.to_datetime('2017-12-31')]
df1719 = df.copy()
# 2020
df = pd.read_excel(
    r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\Covid\Data\Original\Pollution\MI_composizione_2020+Schivenoglia.xlsx",
    sheet_name='MI_Pascal PM10_IC', header=7)
df = df[list(df.columns[3:4]) + ['PM', 'NO3-', 'SO42-', 'NH4+']]
df.rename(columns={df.columns[0]: 'date'}, inplace=True)
df = df.drop(0)
df['date'] = pd.to_datetime(df['date'])
for col in ['PM', 'NO3-', 'SO42-', 'NH4+']:
    df.loc[df[col].astype(str).str.contains('---'), col] = np.nan
    df.loc[df[col].astype(str).str.contains('<'), col] = 0
    df[col] = pd.to_numeric(df[col])
df20 = df.copy()
# df = pd.concat([df1719, df20])

df = df.loc[df['date'] < pd.to_datetime('2020-05-05')]
df = df.loc[df['date'] > pd.to_datetime('2020-01-01')]
ammonium_nitrates, ammonium_sulfates = secondary_pm(df['NO3-'].values, df['SO42-'].values, df['NH4+'].values)
df['an'], df['as'] = ammonium_nitrates, ammonium_sulfates
df['totn'] = df['an'] + df['as']
df['ratio_n2pm'] = (df['totn']) / df['PM']
df['ratio_an2pm'] = (df['an']) / df['PM']
df['ratio_as2pm'] = (df['as']) / df['PM']

fig, ax = plt.subplots(figsize=(12, 4))
fontsize = 12
ax.stackplot(df['date'], df['an'], df['as'], df['PM'] - (df['an'] + df['as']),
             colors=['tab:blue', 'tab:orange', 'gainsboro'],
             labels=['Ammonium nitrate', 'Ammonium sulfate', 'Primary PM$_{10}$ and other secondary PM$_{10}$'])
format_ticks(ax)
vertical_lines(ax)
rotate_ticks()
place_text(ax, fontsize=fontsize)
ax.set_ylabel('$\mu g/m^3$', fontsize=fontsize)
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), reverse=True))
ax.legend(handles, labels, fontsize=fontsize)
plt.tight_layout()
plt.savefig(context.projectpath() + '/Docs/img/nitrates_to_pm.eps')
plt.savefig(
    r"C:\Users\Granella\Dropbox (CMCC)\Apps\Overleaf\Lockdown & Pollution ERL - Revision\Docs\img\nitrates_to_pm.eps")
plt.show()

df.loc[df['date'] > pd.to_datetime('2020-02-21'), 'totn'].describe()
