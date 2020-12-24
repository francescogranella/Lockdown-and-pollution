import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.formula.api as smf
from tqdm import tqdm

import context
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

########################################################################################################################

########################################################################################################################
# Estimate effect of lockdown on pollution with OLS
df = pd.read_parquet(context.projectpath() + '/Data/Modified/RegressionData.parq')
df.drop(columns=['pollutant', 'nomestazione', 'datastart', 'datastop'], inplace=True)
df = pd.merge(df, pollution_metadata, on='idsensore', validate='m:1')
df = df[df['dt'] < pd.to_datetime('2020-05-04')]

df['post'] = df['dt'] > pd.to_datetime('2020-02-21')

regdata = df[
    ['idsensore', 'val', 'prec_sum', 'temp_mean', 'winddir_mean', 'windspeed_mean', 'pollutant', 'type', 'post']]
regdata.dropna(how='any', inplace=True)

no2 = regdata[regdata['pollutant'] == 'Biossido di Azoto']
pm25 = regdata[regdata['pollutant'] == 'Particelle sospese PM2.5']


formula = 'val ~ post + prec_sum + temp_mean + winddir_mean + windspeed_mean + C(pollutant) + C(type)'
model = smf.ols(formula=formula, data=no2)
res = model.fit(cov_type='cluster', cov_kwds={'groups': no2['idsensore'].values}, use_t=True)
print(res.summary())


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mean_bias = np.mean(y_true - y_pred)
    nmean_bias = mean_bias / np.mean(y_true)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    crmse = mean_squared_error(y_true, y_pred - np.mean(y_pred) + np.mean(y_true), squared=False)
    ncrmse = crmse / np.mean(y_true)
    pcc, _ = pearsonr(y_true, y_pred)
    return dict(r2=r2, mean_bias=mean_bias, nmean_bias=nmean_bias, rmse=rmse, crmse=crmse, ncrmse=ncrmse, pcc=pcc)


groups = df.groupby('idsensore')
results_list = []
for i, group in tqdm(groups):
    group = group[['dt', 'idsensore', 'pollutantshort', 'pop', 'val', 'prec_sum', 'temp_max', 'temp_min', 'temp_mean',
                   'winddir_mean', 'windspeed_mean']].dropna(axis=0, how='any')

    Xcols = ['prec_sum', 'temp_mean', 'winddir_mean', 'windspeed_mean']
    Xdummies = pd.concat(
        [pd.get_dummies(pd.qcut(group[x], q=5, labels=False, duplicates='drop'), prefix=x) for x in Xcols], axis=1)
    Xcols = list(Xdummies.columns) + ['prec_sum']
    group = pd.concat([group, Xdummies], axis=1)

    train = group[group.dt.dt.year < 2020]
    test = group[group.dt.dt.year == 2020]

    if len(train) == 0 or len(test) == 0:
        continue

    # Split train and test
    y_train, X_train = train['val'], train[Xcols]
    X_train = sm.add_constant(X_train)
    y_test, X_test = test['val'], test[Xcols]
    X_test = sm.add_constant(X_test)

    # Fit the model
    model = sm.OLS(y_train, X_train)
    model = model.fit()

    # Evaluate
    # In-sample prediction
    train_pred = model.predict(X_train)
    train_metrics = pd.DataFrame(metrics(y_train, train_pred), index=[0])
    train_metrics.columns = ['train_' + x for x in train_metrics.columns]

    # Out-of-sample prediction
    test_pred = model.predict(X_test)
    test_metrics = pd.DataFrame(metrics(y_test, test_pred), index=[0])
    test_metrics.columns = ['test_' + x for x in test_metrics.columns]

    idsensore = group['idsensore'].iloc[0]
    idsensore_df = pd.DataFrame(dict(idsensore=idsensore), index=[0])
    train_test_metrics = pd.concat([idsensore_df, train_metrics, test_metrics], axis=1)

    # Store predictions
    test_pred.name = 'y_pred'
    res = pd.concat([test[['dt', 'idsensore', 'pollutantshort', 'pop']], y_test, test_pred], axis=1)
    res = pd.merge(res, train_test_metrics, on='idsensore', how='inner')

    results_list.append(res)

results = pd.concat(results_list)

no2 = results[(results['pollutantshort'] == 'NO2') & (results['dt'] < pd.to_datetime('2020-02-22'))]
pm25 = results[(results['pollutantshort'] == 'PM2.5') & (results['dt'] < pd.to_datetime('2020-02-22'))]
descriptives_no2 = no2[[x for x in no2.columns if 'test' in x]].describe()
descriptives_pm25 = pm25[[x for x in pm25.columns if 'test' in x]].describe()

from tabulate import tabulate

print(tabulate(descriptives_no2, headers='keys', tablefmt='grid'))
print(tabulate(descriptives_pm25, headers='keys', tablefmt='grid'))

no2_test = pd.DataFrame(no2[[x for x in no2.columns if 'test' in x]].describe().iloc[1]).T
no2_test.columns = [x.replace('test_', '') for x in no2_test.columns]
no2_test['Dataset'] = 'Test'
no2_test.index = ['NO2']
no2_train = pd.DataFrame(no2[[x for x in no2.columns if 'train' in x]].describe().iloc[1]).T
no2_train.columns = [x.replace('train_', '') for x in no2_train.columns]
no2_train['Dataset'] = 'Train'
no2_train.index = ['NO2']
pm25_test = pd.DataFrame(pm25[[x for x in pm25.columns if 'test' in x]].describe().iloc[1]).T
pm25_test.columns = [x.replace('test_', '') for x in pm25_test.columns]
pm25_test['Dataset'] = 'Test'
pm25_test.index = ['PM2.5']
pm25_train = pd.DataFrame(pm25[[x for x in pm25.columns if 'train' in x]].describe().iloc[1]).T
pm25_train.columns = [x.replace('train_', '') for x in pm25_train.columns]
pm25_train['Dataset'] = 'Train'
pm25_train.index = ['PM2.5']

desc = pd.concat([no2_train, no2_test, pm25_train, pm25_test], axis=0)
desc = desc[['Dataset', 'pcc', 'mean_bias', 'nmean_bias', 'rmse', 'crmse', 'ncrmse']]
desc.index.name = 'Pollutant'
desc = np.round(desc, 2)
desc.columns = ['Dataset', 'Corr', 'MB', 'nMB', 'RMSE', 'cRMSE', 'ncRMSE']
desc_table = tabulate(desc, tablefmt='latex_booktabs', headers='keys')
desc_table_notes = r"""
\begin{flushleft}
\footnotesize\textit{Notes: } \textit{Corr}: Pearson's correlation coefficient. \textit{MB}: Mean bias, where negative values indicate observed values below predicted values. \textit{nMB}: Normalized mean bias. \textit{RMSE}: Root mean squared error. \textit{nRMSE}: Normalized RMSE. \textit{cRMSE}: Centered RMSE. \textit{ncRMSE}: Normalized centered RMSE. Mean bias, RMSE and centered RMSE are expressed in $\mu g/m^3$. Mean bias, RMSE and centered RMSE are normalized dividing by mean observed concentrations. The centered RMSE is computed as $\big[ 1/N \sum (\widehat{y}_i  -\bar{\widehat{y}} - y_i + \bar{y})^2 \big]^{1/2}$.
\end{flushleft}"""
with open(context.projectpath() + '/Docs/tables/evaluation_OLS.tex', 'w') as f:
    f.write(desc_table + desc_table_notes)
print(desc_table + desc_table_notes)

# Average values, weighted by population
wm = lambda x: np.average(x, weights=x["pop"])
Observed = results.groupby(['pollutantshort', 'dt']).apply(lambda x: np.average(x['val'], weights=x['pop']))
Observed.name = 'Observed'
Predicted = results.groupby(['pollutantshort', 'dt']).apply(lambda x: np.average(x['y_pred'], weights=x['pop']))
Predicted.name = 'Predicted'
wgt_avg = pd.concat([Observed, Predicted], axis=1).reset_index()

########################################################################################################################
# GRAPHS
import matplotlib.dates as mdates

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
    if poll == 'PM2.5':
        ylims = (0, 80)
    ax.set_ylim(ylims)
    max = ylims[1]
    fontsize = 12
    ax.text(codogno + pd.DateOffset(offset), max * 0.85, "           Lockdown\n11 municipalities", fontsize=fontsize)
    ax.text(schools_closed, max * 0.85, " Schools\n close", fontsize=fontsize)
    ax.text(lockdown, max * 0.85, " Lockdown\n Lombardy", fontsize=fontsize)


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
    plt.savefig(context.projectpath() + f'/Docs/img/Avg_values_{poll}_OLS.eps', dpi=200)
    plt.show()
