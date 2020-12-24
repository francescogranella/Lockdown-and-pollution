"""
Evaluate results of prediction
"""

import glob
from datetime import datetime

import context
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tabulate import tabulate
from utils import pop_municipalities

context.pdsettings()

########################################################################################################################
# METADATA
pollution_metadata = pd.read_parquet(context.projectpath() + '/Data/Modified/PollutionStations.parq')
pollution_metadata = pollution_metadata[pollution_metadata['comune'] != 'Galliate']  # Monitor just outside Lombardy
pollution_metadata = pollution_metadata[pollution_metadata['comune'] != 'Salionze']  # Monitor just outside Lombardy
pollution_metadata = pollution_metadata[pollution_metadata['comune'] != 'Melara']  # Monitor just outside Lombardy
pollution_metadata = pollution_metadata[pollution_metadata['comune'] != 'Ceneselli']  # Monitor just outside Lombardy

# Monitor type (background, industrial, traffic)
airbase = pd.read_csv('http://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv', sep='\t')
airbase = airbase.loc[airbase['Countrycode'] == 'IT']
airbase.reset_index(drop=True, inplace=True)
tree = cKDTree(airbase[['Longitude', 'Latitude']].values)
dist, idx = tree.query(pollution_metadata[['lng', 'lat']].values, k=1)
pollution_metadata['type'], pollution_metadata['area'] = zip(
    *airbase.iloc[idx][['AirQualityStationType', 'AirQualityStationArea']].values)
# pollution_metadata['dist'] = dist

# Pop density
pop = pop_municipalities()
print(pop.head())

pollution_metadata = pd.merge(pollution_metadata, pop, on='comune', how='left', indicator='indicator')
assert pollution_metadata['indicator'].unique() == 'both'
pollution_metadata.drop(columns='indicator', inplace=True)

# Predictions
prediction_files = glob.glob(context.projectpath() + f'/Data/Modified/Predictions/Predictions *.parq')
all_predictions = pd.concat([pd.read_parquet(x) for x in prediction_files], axis=0)
all_predictions['months'] = all_predictions['months'].transform(lambda x: str(x))
all_predictions.drop_duplicates(inplace=True)
all_predictions.reset_index(drop=True, inplace=True)

metrics = all_predictions.drop(columns=['dt', 'Observed', 'Predicted'])
metrics['months'] = metrics['months'].transform(lambda x: str(x))
metrics = metrics.drop_duplicates()
metrics.sort_values(by=['idsensore', 'test_crmse'])

metrics = pd.merge(metrics, pollution_metadata[['idsensore', 'pollutantshort', 'type', 'pop']], how='left',
                   on='idsensore')

# Best prediction: smallest CRMSE
idx = all_predictions.groupby(['idsensore', 'dt'])['test_crmse'].idxmin()
predictions = all_predictions.loc[idx]
predictions.sort_values(by=['idsensore', 'test_crmse'])

predictions = pd.merge(predictions, pollution_metadata[['idsensore', 'pollutantshort', 'type', 'pop']], how='left',
                       on='idsensore')
no2 = predictions[predictions['pollutantshort'] == 'NO2']
pm25 = predictions[predictions['pollutantshort'] == 'PM2.5']

# Summarize metrics
descriptives = metrics[[x for x in metrics.columns if 'test' in x]].describe()

descriptives_no2 = metrics.loc[
    metrics['pollutantshort'] == 'NO2', [x for x in metrics.columns if 'test' in x]].describe()
descriptives_pm25 = metrics.loc[
    metrics['pollutantshort'] == 'PM2.5', [x for x in metrics.columns if 'test' in x]].describe()

print(tabulate(descriptives_no2, headers='keys', tablefmt='grid'))
print(tabulate(descriptives_pm25, headers='keys', tablefmt='grid'))

########################################################################################################################
# REGRESSIONS
no2.to_stata(context.projectpath() + '/Data/Modified/NO2.dta', write_index=False)
pm25.to_stata(context.projectpath() + '/Data/Modified/PM2.5.dta', write_index=False)

########################################################################################################################
# GRAPHS

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


# ---------------
# Mobility graph

# Google
# Open
# google = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=9cfdda8c20841ad1')
google = pd.read_csv(context.projectpath() + '/Data/Original/Mobility/Global_Mobility_Report.csv')
google = google.loc[google['sub_region_1'] == 'Lombardy'].drop(
    columns=['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2']).set_index('date')
google.rename(columns={'transit_stations_percent_change_from_baseline': 'transit'}, inplace=True)
# Rescale
google = (google - google.loc[google.index == '2020-02-23'].values) + 100
google.index = pd.to_datetime(google.index)

# Apple
# Open
# https://covid19-static.cdn-apple.com/covid19-mobility-data/2017HotfixDev13/v3/en-us/applemobilitytrends-2020-09-22.csv
apple = pd.read_csv(context.projectpath() + '/Data/Original/Mobility/applemobilitytrends-2020-09-22.csv')
apple.drop(columns=['geo_type', 'alternative_name', 'sub-region', 'country'], inplace=True)
apple = apple[apple['region'] == 'Lombardy Region'].drop(columns='region').set_index('transportation_type', ).T
apple.index = pd.to_datetime(apple.index, format='%Y-%m-%d')
apple.index.name = 'date'
# Cut time series
apple = apple[apple.index < pd.to_datetime('2020-05-08', format='%Y-%m-%d')]
# Rescale
apple = (apple - apple.loc[apple.index == '2020-02-23'].values) + 100

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
fontsize = 12

ax.plot(google.index, google['transit'], label='Transit (Google)')
ax.plot(apple.index, apple['driving'], label='Transit (Apple)')
ax.plot(apple.index, apple['walking'], label='People walking (Apple)')

ax.set_ylim((0, 210))
format_ticks(ax)
vertical_lines(ax)
rotate_ticks()
place_text(ax, fontsize=fontsize)

# Axis label
ax.set_ylabel(f'Mobility index (23/02=100)', fontsize=fontsize)

# Legend
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, fontsize=fontsize)
# Axis title
ax.set_title(f'Mobility in Lombardia', fontsize=fontsize)

plt.tight_layout()
plt.savefig(context.projectpath() + '/Docs/img/mobility.eps', dpi=700)
plt.show()

# ---------------
# TERNA
# 2020
df20 = pd.read_excel(context.projectpath() + '/Data/Original/Mobility/Terna_TotalLoad2020.xlsx', skiprows=2)
df20.columns = ['dt', 'tot', 'forecast', 'zone']
df20['dt'] = df20['dt'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour))
df20 = df20.groupby(['dt'])['tot'].mean().reset_index()
df20.columns = ['dt20', 'tot20']

# 2019
df19 = pd.read_excel(context.projectpath() + '/Data/Original/Mobility/Terna_TotalLoad2019.xlsx', skiprows=2)
df19.columns = ['dt', 'tot', 'forecast', 'zone']
df19['dt'] = df19['dt'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour))
df19 = df19.groupby(['dt'])['tot'].mean().reset_index()
df19.columns = ['dt19', 'tot19']
# Shift 2019 by 1 day to match day-of-week
df19['tot19'] = df19['tot19'].shift(-24)

# Adjust time series to match day of the week
dates_range = pd.date_range(df20['dt20'].min(), df20['dt20'].max(), freq='h')
df20['dt'] = df20['dt20'].apply(lambda x: datetime(year=x.year, month=x.month, day=x.day, hour=x.hour))
df19['dt'] = df19['dt19'].apply(lambda x: datetime(year=2020, month=x.month, day=x.day, hour=x.hour))

df20 = df20.set_index('dt').reindex(dates_range).reset_index().rename(columns={'index': 'dt'})
df19 = df19.set_index('dt').reindex(dates_range).reset_index().rename(columns={'index': 'dt'})

# Merge
df = pd.merge(df20, df19, on='dt')

# Shift 2019 by 1 day to match day-of-week (b/c of Feb 29)
df.loc[df['dt'] > pd.to_datetime('20200229'), 'tot19'] = df.loc[df['dt'] > pd.to_datetime('20200229'), 'tot19'].shift(
    -24)

df['date'] = df['dt'].apply(lambda x: datetime(year=x.year, month=x.month, day=x.day))

pdf = df.loc[(df['date'] >= pd.to_datetime('20200201')) & (df['date'] < pd.to_datetime('20200331'))]

fig, ax = plt.subplots(figsize=(14, 5))
fontsize = 12

ax.plot(pdf['dt'], pdf['tot19'], label='2019')
ax.plot(pdf['dt'], pdf['tot20'], label='2020')
ax.set_ylim((0, ax.get_ylim()[1] * 1.1))
format_ticks(ax)
vertical_lines(ax)
rotate_ticks()
place_text(ax, fontsize=fontsize, offset=-8)

# Labels
ax.set_ylabel('Total load, MW', fontsize=fontsize)
plt.xticks(rotation=90, ha='center', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.title('Total load MW, North bidding zone', fontsize=fontsize)
plt.tight_layout()
plt.savefig(context.projectpath() + '/Docs/img/TernaTotalLoad.eps', dpi=200)
plt.show()

# ---------------
# Average values, weighted by population
wm = lambda x: np.average(x, weights=predictions.loc[x.index, "pop"])
wgt_avg = predictions.groupby(['pollutantshort', 'dt']).agg(Observed=('Observed', wm),
                                                            Predicted=('Predicted', wm)).reset_index()

for poll in ['PM2.5', 'NO2']:
    plotdf = wgt_avg[wgt_avg['pollutantshort'] == poll]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plotdf['dt'], plotdf['Observed'], label='Observed')
    ax.plot(plotdf['dt'], plotdf['Predicted'], label='Predicted')

    format_ticks(ax)
    vertical_lines(ax)
    rotate_ticks()
    place_text(ax)

    plt.title(poll)
    plt.legend()
    poll = poll.replace('.', '')
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/Avg_values_{poll}.eps', dpi=200)
    plt.show()

# ---------------
# Taylor graph
import skill_metrics as sm


def make_stats(df):
    corr, crmsd, stdev, ids, comuni = [1], [0], [1], ['ref'], ['lom']
    n = 0
    for i, group in df.groupby('idsensore'):
        stats = sm.taylor_statistics(group['Predicted'], group['Observed'])
        corr.append(stats['ccoef'][1])
        crmsd.append(stats['crmsd'][1])
        stdev.append(stats['sdev'][1] / stats['sdev'][0])
        n += 1
        ids.append(str(n))
    corr = np.array(corr)
    crmsd = np.array(crmsd)
    stdev = np.array(stdev)
    label = dict(zip(ids, comuni))
    return corr, crmsd, stdev, ids


for poll in ['PM2.5', 'NO2']:
    plotdf = predictions.loc[
        (predictions['pollutantshort'] == poll) & (predictions['dt'] < pd.to_datetime('2020-02-23'))]
    traffic = plotdf.loc[plotdf['type'] == 'traffic']
    bg = plotdf.loc[plotdf['type'] == 'background']
    ind = plotdf.loc[plotdf['type'] == 'industrial']

    tcorr, tcrmsd, tstdev, tids = make_stats(traffic)
    bcorr, bcrmsd, bstdev, bids = make_stats(bg)
    icorr, icrmsd, istdev, iids = make_stats(ind)

    tids = ['ref'] + [str(x) for x in range(1, len(tcorr))]
    bids = ['ref'] + [str(x) for x in range(len(tcorr), len(tcorr) + len(bcorr))]
    iids = ['ref'] + [str(x) for x in range(len(tcorr) + len(bcorr), len(tcorr) + len(bcorr) + len(icorr))]

    label = {'Traffic': 'r', 'Background': 'b', 'Industrial': 'g'}

    sm.taylor_diagram(tstdev, tcrmsd, tcorr, markercolor='r', alpha=0.0,
                      colCOR='black', styleCOR='dotted', widthCOR=0.75,
                      styleSTD='dotted', widthSTD=0.75,
                      markerSize=10,
                      # tickRMS=[0, 0.25, 0.5, 0.75, 1],
                      # colRMS='black', titleRMSDangle=120,
                      # tickRMSangle=175,
                      showlabelsRMS='off',
                      titleRMS='off', widthRMS=0,
                      styleOBS='-', colOBS='r', markerobs='D',
                      titleOBS='Perfect prediction', widthOBS=0,
                      checkstats='on', markerlabel=label, markerlegend='off')

    sm.taylor_diagram(bstdev, bcrmsd, bcorr, markercolor='b', alpha=0.0,
                      overlay='on',
                      markerSize=10)
    sm.taylor_diagram(istdev, icrmsd, icorr, markercolor='g', alpha=0.0,
                      overlay='on',
                      markerSize=10)
    plt.title(poll)
    poll = poll.replace('.', '')
    plt.savefig(context.projectpath() + f'/Docs/img/Taylor_{poll}.eps', dpi=100)
    plt.show()
