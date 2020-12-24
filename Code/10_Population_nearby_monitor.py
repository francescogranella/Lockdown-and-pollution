import glob

import context
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geovoronoi import voronoi_regions_from_coords
from scipy.spatial import cKDTree

context.pdsettings()

# Set CRS
crs = 'EPSG:3035'


def lombardia():
    reg = gpd.read_file(
        context.projectpath() + "/Data/Original/Shapefiles/Limiti01012020/Reg01012020/Reg01012020_WGS84.shp")
    lomb = reg[reg['DEN_REG'] == 'Lombardia']
    lomb = lomb.to_crs(crs)
    return lomb


def comuni():
    # Population
    pop = pd.read_csv(
        context.projectpath() + "/Data/Original/Population/Elenco-codici-statistici-e-denominazioni-al-01_01_2020.csv",
        encoding='latin-1', sep=';', decimal=',', thousands='.')
    pop = pop.loc[pop['Denominazione regione'] == 'Lombardia', [
        "Denominazione dell'Unità territoriale sovracomunale \n(valida a fini statistici)", 'Denominazione in italiano',
        'Codice Comune formato numerico', 'Popolazione legale 2011 (09/10/2011)']]
    pop.columns = ['prov', 'comune', 'code', 'pop']
    # Comuni
    comuni = gpd.read_file(
        context.projectpath() + "/Data/Original/Shapefiles/Limiti01012020/Com01012020/Com01012020_WGS84.shp")
    comuni = comuni.loc[comuni.COD_REG == 3]
    comuni = comuni.to_crs(crs)
    # Assign population to comuni
    comuni = comuni.merge(pop, how='outer', left_on='PRO_COM', right_on='code', indicator='m', validate='1:1')
    assert comuni.m.unique() == ['both']
    comuni = comuni[['prov', 'code', 'comune', 'pop', 'geometry']]
    return comuni


def monitors():
    # Metadata
    meta = pd.read_stata(context.projectpath() + '/Data/Modified/PollutionStations.dta')
    # Type of monitor
    airbase = pd.read_csv(context.projectpath() + '/Data/Original/Pollution/PanEuropean_metadata.csv', sep='\t')
    airbase = airbase.loc[airbase['Countrycode'] == 'IT']
    airbase.reset_index(drop=True, inplace=True)
    tree = cKDTree(airbase[['Longitude', 'Latitude']].values)
    dist, idx = tree.query(meta[['lng', 'lat']].values, k=1)
    meta['type'], meta['area'] = zip(*airbase.iloc[idx][['AirQualityStationType', 'AirQualityStationArea']].values)
    # Monitors in study
    outputs = glob.glob(context.projectpath() + '/Data/Modified/Output *.dta')
    dflist = []
    for x in outputs:
        dflist.append(pd.read_stata(x))
    temp = pd.concat(dflist, sort=False)
    ids = temp['idsensore'].unique()
    monitors = meta.loc[meta['idsensore'].isin(ids), ['idsensore', 'pollutantshort', 'lng', 'lat', 'type']]
    geom = gpd.points_from_xy(monitors['lng'], monitors['lat'])
    monitors = gpd.GeoDataFrame(monitors, geometry=geom, crs='EPSG:4326')
    monitors = monitors.to_crs(crs)
    return monitors


def ordered_voronoi(pollutantshort, type):
    df = monitors.loc[(monitors['pollutantshort'] == pollutantshort) & (monitors['type'] == type)]
    # Monitor identifiers
    ids = df['idsensore']
    # Monitor coordinates
    coords = df['geometry'].transform(lambda x: x.coords[0])
    coords = coords.to_list()
    # Voronoi calculation fails with 2 points. Add an artificial one outside Italy
    if len(df) == 2:
        coords.append((0, 0))
    coords = np.array(coords)
    # Double-check
    assert df.crs == lomb.crs
    # Outer boundary of Voronoi: Lombardia
    shape = lomb.iloc[0].geometry
    # Voronoi
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, shape)
    if len(df) == 2:
        del pts[-1]
    # Order Voronoi polygons and monitor points
    order = [x for sublist in poly_to_pt_assignments for x in sublist]
    ordered_poly_shapes = [x for _, x in sorted(zip(order, poly_shapes))]
    return ordered_poly_shapes, pts


def truncated_buffers(pollutantshort, type, ordered_poly_shapes, pts, radius=0.2, plot=False):
    df = monitors.loc[(monitors['pollutantshort'] == pollutantshort) & (monitors['type'] == type)]
    # Check
    assert len(df) == len(ordered_poly_shapes)
    # Truncated buffers: intersection between buffers around monitors and Voronois
    trbufferlist = []
    for i in range(0, len(df)):
        trbuffer = pts[i].buffer(radius).intersection(ordered_poly_shapes[i])
        trbufferlist.append(trbuffer)
    # Plot
    if plot:
        gpd.GeoSeries(trbufferlist).plot(alpha=0.5, edgecolor='k', cmap='tab10')
        plt.show()
    return trbufferlist


def overlay(pollutantshort, type, popgrid, truncated_buffers, crs=crs, plot=False):
    df = monitors.loc[(monitors['pollutantshort'] == pollutantshort) & (monitors['type'] == type)]
    df = gpd.GeoDataFrame(df.drop(columns=['geometry', 'lng', 'lat']), geometry=truncated_buffers, crs=crs)
    popgrid['areacomune'] = popgrid.geometry.area
    truncated_buffers = gpd.GeoSeries(truncated_buffers).boundary
    # Overlay: every row is the intersection of a comune and a truncated buffer around monitor
    df = gpd.overlay(popgrid, df, how='intersection')
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        lomb.boundary.plot(ax=ax, color='black')
        truncated_buffers.plot(ax=ax, color='black')
        extent = lomb.total_bounds  # + [-1,-1,1,1]
        gplt.choropleth(df, ax=ax, alpha=1, hue='pop', cmap='Reds', edgecolor=None, linewidth=0.00001, legend=True,
                        extent=extent)
        plt.tight_layout()
        plt.savefig(context.projectpath() + f'/Docs/img/monitor_radii_{pollutantshort}_{type}.eps', transparent=False,
                    dpi=200)
        plt.close()
    return df, truncated_buffers


def monitor_pop(df):
    # Share of population of a comuni within the buffer. Assumes homogenoues density within a comune
    df['areainter'] = df.geometry.area
    df['share_area_im'] = df['areainter'] / df['areacomune']
    df['pop'] = df['pop'] * df['share_area_im']
    # Population within the buffer
    return df.groupby('idsensore')['pop'].sum().reset_index()


comuni = comuni()
lomb = lombardia()
monitors = monitors()

popgrid = gpd.read_file(
    context.projectpath() + r"/Data/Original/Shapefiles/GEOSTAT_grid_POP_1K_IT_2011-22-10-2018/GEOSTAT_grid_POP_1K_IT_2011.shp",
    mask=lomb)
popgrid = popgrid.to_crs(crs)
popgrid.rename(columns={'Pop': 'pop'}, inplace=True)

# Population within a given radius
poplist = []
shplist = []
tbblist = []

# Radius in meters
radius = 20000
for pollutant in ['PM2.5', 'NO2']:
    for monitor_type in monitors['type'].unique():
        print(pollutant, monitor_type)
        ov, pts = ordered_voronoi(pollutant, monitor_type)
        tb = truncated_buffers(pollutant, monitor_type, ov, pts, radius=radius, plot=False)
        shp, tbboundary = overlay(pollutant, monitor_type, popgrid, tb, plot=False)
        temp = monitor_pop(shp)
        poplist.append(temp)
        shplist.append(shp)
        tbblist.append(tbboundary)

# Export

pd.concat(poplist).to_csv(context.projectpath() + '/Data/Modified/MonitorsPop.csv', index=False)

types = [x.title() for x in monitors['type'].unique()]

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
lomb = lomb.to_crs(epsg=4326)
for i in range(0, 3):
    # PM 2.5
    tb = gpd.GeoSeries(tbblist[i], crs=crs).to_crs(epsg=4326)
    shp = shplist[i].to_crs(epsg=4326)

    lomb.boundary.plot(ax=axs[i, 0], color='black')
    tb.plot(ax=axs[i, 0], color='black')
    extent = lomb.total_bounds
    gplt.choropleth(shp, ax=axs[i, 0], alpha=1, hue='pop', cmap='Reds', edgecolor='none', linewidth=None, legend=True,
                    extent=extent)
    # NO2
    tb = gpd.GeoSeries(tbblist[i + 3], crs=crs).to_crs(epsg=4326)
    shp = shplist[i + 3].to_crs(epsg=4326)

    axs[i, 0].set_title(f'PM 2.5 - {types[i]}')
    lomb.boundary.plot(ax=axs[i, 1], color='black')
    tb.plot(ax=axs[i, 1], color='black')
    extent = lomb.total_bounds
    gplt.choropleth(shp, ax=axs[i, 1], alpha=1, hue='pop', cmap='Reds', edgecolor='none', linewidth=None, legend=True,
                    extent=extent)
    axs[i, 1].set_title(f'NO2 - {types[i]}')
plt.tight_layout()
plt.savefig(context.projectpath() + f'/Docs/img/monitor_radii.pgf', dpi=100)
plt.show()
