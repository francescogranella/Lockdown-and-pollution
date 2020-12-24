import context
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

context.pdsettings()

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

lombardia = gpd.read_file(context.projectpath() + "/Data/Original/Shapefiles/ITA_adm1.shp")
lomb = lombardia[lombardia['NAME_1'] == 'Lombardia']

# Pop grid in lombardia
popgrid = gpd.read_file(
    context.projectpath() + r"/Data/Original/Shapefiles/GEOSTAT_grid_POP_1K_IT_2011-22-10-2018/GEOSTAT_grid_POP_1K_IT_2011.shp",
    mask=lomb)
popgrid = popgrid.to_crs(lomb.crs)
popgrid['lpop'] = np.log(popgrid['Pop'] + 1)


def empty_ticks(ax):
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])


def plot_boundaries(ax):
    lomb.boundary.plot(ax=ax, edgecolor='black', zorder=0)


def pop_grid(ax):
    pop_alpha = 0.5
    popgrid.plot(ax=ax, column='Pop', alpha=pop_alpha, cmap='Greys',
                 norm=LogNorm(vmin=1, vmax=popgrid['Pop'].max()))


for pollutant in ['NO2', 'PM2.5']:
    df = pd.read_stata(context.projectpath() + f'/Data/Modified/{pollutant}.dta')
    pollutant = pollutant.replace('.', '')

    df['diff'] = df['Observed'] - df['Predicted']
    df['post'] = (df['dt'] > pd.to_datetime('2020-02-21'))
    df['adjustment'] = df.groupby(['idsensore', 'post'])['diff'].transform('mean')
    df['adjustment'][df['post'] == 1] = np.nan
    df['adjustment'] = df.groupby('idsensore')['adjustment'].transform(np.nanmean)
    df['diff'] = df['diff'] - df['adjustment']

    df = df[df['dt'] > pd.to_datetime('2020-02-23')]
    df['biweek'] = df['dt'].dt.week // 2 - np.min(df['dt'].dt.week // 2)

    from_to = df.groupby('biweek')['dt'].agg(['min', 'max'])
    from_to['min'] = from_to['min'].dt.strftime('%b %d')
    from_to['max'] = from_to['max'].dt.strftime('%b %d %Y')
    from_to = from_to.astype(str).agg(' to '.join, axis=1).to_dict()

    df = df.groupby(['idsensore', 'biweek'])['Observed', 'Predicted', 'diff'].mean().reset_index()
    df['diff'] = df['Observed'] - df['Predicted']
    df = pd.merge(df, pollution_metadata[['idsensore', 'type', 'lat', 'lng']], how='left', on='idsensore')
    df = df[df['type'] == 'background']
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')

    import matplotlib.colors as mcolors

    nweek = df['biweek'].unique()

    # Observed
    var = 'Observed'
    max = np.round(df[var].max(), -1)
    min = np.round(df[var].min(), -1)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        empty_ticks(ax)
        plot_boundaries(ax)
        pop_grid(ax)
        temp = df[df['biweek'] == nweek[i]]
        temp.plot(column=var, ax=ax, vmin=min, vmax=max, legend=True, cmap='Reds')
        ax.axis('off')
        ax.set_title(from_to[nweek[i]])
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/spatial_observed_{pollutant}.eps')
    plt.savefig(
        rf'C:\Users\Granella\Dropbox (CMCC)\Apps\Overleaf\Lockdown & Pollution ERL - Revision/Docs/img/spatial_observed_{pollutant}.eps')
    plt.close()
    # plt.show()

    var = 'Predicted'
    max = np.round(df[var].max(), -1)
    min = np.round(df[var].min(), -1)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        empty_ticks(ax)
        plot_boundaries(ax)
        pop_grid(ax)
        temp = df[df['biweek'] == nweek[i]]
        temp.plot(column=var, ax=ax, vmin=min, vmax=max, legend=True, cmap='Reds')
        ax.axis('off')
        ax.set_title(from_to[nweek[i]])
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/spatial_predicted_{pollutant}.eps')
    plt.savefig(
        rf'C:\Users\Granella\Dropbox (CMCC)\Apps\Overleaf\Lockdown & Pollution ERL - Revision/Docs/img/spatial_predicted_{pollutant}.eps')
    plt.close()
    # plt.show()

    # DIFF
    var = 'diff'
    max = np.round(df[var].max(), -1)
    min = np.round(df[var].min(), -1)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        empty_ticks(ax)
        plot_boundaries(ax)
        pop_grid(ax)
        temp = df[df['biweek'] == nweek[i]]
        temp.plot(column=var, ax=ax, vmin=min, vmax=max, legend=True, cmap='seismic',
                  norm=mcolors.DivergingNorm(vmin=min, vcenter=0, vmax=max))
        ax.axis('off')
        ax.set_title(from_to[nweek[i]])
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/spatial_dif_{pollutant}.eps')
    plt.savefig(
        rf'C:\Users\Granella\Dropbox (CMCC)\Apps\Overleaf\Lockdown & Pollution ERL - Revision/Docs/img/spatial_dif_{pollutant}.eps')
    plt.close()
    # plt.show()


# ======================================================================================
# Spatial disitrubution of health benefits

def mort_1person(delta, rr):
    from io import StringIO
    mortality_table = """age	le_females	le_males	p_males	p_females	male	female
    30	53.647	49.5125	.00054245	.00031985	513897	482095
    40	43.843	39.805	.00127031	.00081897	614984	601969
    50	34.255	30.379	.00357404	.00221435	820037	800031
    60	25.055	21.545	.00986602	.00527062	758324	768297
    70	16.4085	13.7105	.02778369	.01538495	561462	611529
    80	8.983	7.357	.09032641	.06118978	435068	528735
    90	4.122	3.407	.25322081	.20329991	195999	331496
    100	1.877	1.598	.53461663	.46866033	21816	73133
    """
    mortality_table = pd.read_csv(StringIO(mortality_table), sep='\t')
    tot = mortality_table['male'].sum() + mortality_table['female'].sum()
    mortality_table['male'] = mortality_table['male']
    mortality_table['female'] = mortality_table['female']

    beta = np.log(rr) / 10

    mortality_table['avoided_deaths_female'] = mortality_table['p_females'] * (1 - 1 / (np.exp(beta * delta))) / 6 * \
                                               mortality_table['female']
    mortality_table['avoided_deaths_male'] = mortality_table['p_males'] * (1 - 1 / (np.exp(beta * delta))) / 6 * \
                                             mortality_table['male']

    return (mortality_table['avoided_deaths_female'].sum() + mortality_table['avoided_deaths_male'].sum()) / tot


for pollutant in ['NO2', 'PM2.5']:
    df = pd.read_stata(context.projectpath() + f'/Data/Modified/{pollutant}.dta')
    pollutant = pollutant.replace('.', '')

    df['diff'] = df['Observed'] - df['Predicted']
    df['post'] = (df['dt'] > pd.to_datetime('2020-02-21'))
    df['adjustment'] = df.groupby(['idsensore', 'post'])['diff'].transform('mean')
    df['adjustment'][df['post'] == 1] = np.nan
    df['adjustment'] = df.groupby('idsensore')['adjustment'].transform(np.nanmean)
    df['diff'] = df['diff'] - df['adjustment']

    df = df[df['dt'] > pd.to_datetime('2020-02-23')]

    df = df.groupby(['idsensore', 'pop'])['diff'].mean().reset_index()
    df = pd.merge(df, pollution_metadata[['idsensore', 'type', 'lat', 'lng']], how='left', on='idsensore')
    df = df[df['type'] == 'background']
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')

    if pollutant == 'PM2.5':
        rr = 1.062
    elif pollutant == 'NO2':
        rr = 1.055

    df['local_avoided_deaths'] = df['diff'].transform(lambda x: mort_1person(x, rr)) * df['pop'] * -1

    from matplotlib.colors import LogNorm

    temp = df.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    empty_ticks(ax)
    plot_boundaries(ax)
    pop_grid(ax)

    temp.plot(ax=ax, alpha=1, markersize=45, edgecolor='k', linewidth=0.2, marker='X', color='red', zorder=2)
    labels = np.round(temp['local_avoided_deaths'], 1)
    for x, y, label in zip(temp.geometry.x, temp.geometry.y, labels):
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points", fontweight='bold', fontsize=9,
                    zorder=1, bbox=bbox_props)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(context.projectpath() + f'/Docs/img/spatial_local_deaths_{pollutant}.eps')
    plt.savefig(
        rf'C:\Users\Granella\Dropbox (CMCC)\Apps\Overleaf\Lockdown & Pollution ERL - Revision/Docs/img/spatial_local_deaths_{pollutant}.eps')
    plt.show()
