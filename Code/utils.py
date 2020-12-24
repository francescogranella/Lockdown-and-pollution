import context
import geopandas as gpd
import pandas as pd
import requests


def pop_municipalities():
    import os
    import zipfile
    # If shapefile not on disk, download, save and unzip
    if not os.path.isfile(
            context.projectpath() + "/Data/Original/Shapefiles/Limiti01012020/Com01012020/Com01012020_WGS84.shp"):
        # DOwnload
        print('Downloading shapefile of municipalities')
        url = 'http://www.istat.it/storage/cartografia/confini_amministrativi/non_generalizzati/Limiti01012020.zip'
        req = requests.get(url)
        shapefiles_directory = context.projectpath() + '/Data/Original/Shapefiles/'
        # Save
        with open(shapefiles_directory + 'Limiti01012020.zip', 'wb') as f:
            f.write(req.content)
        # Unzip
        with zipfile.ZipFile(shapefiles_directory + 'Limiti01012020.zip', 'r') as zip_ref:
            zip_ref.extractall(shapefiles_directory)

    # Pop density
    comuni = gpd.read_file(
        context.projectpath() + "/Data/Original/Shapefiles/Limiti01012020/Com01012020/Com01012020_WGS84.shp")
    comuni = comuni.loc[comuni.COD_REG == 3]
    comuni = comuni[['PRO_COM', 'SHAPE_AREA']]

    # # Add populatoin to comune
    # population_directory = context.projectpath() + '/Data/Original/Population/'
    # population_file = population_directory + 'Elenco-codici-statistici-e-denominazioni-delle-unita-territoriali.zip'
    # if not os.path.isfile(population_file):
    #     url = 'https://www.istat.it/storage/codici-unita-amministrative/Elenco-codici-statistici-e-denominazioni-delle-unita-territoriali.zip'
    #     req = requests.get(url)
    #     # Save
    #     with open(population_file,'wb') as f:
    #         f.write(req.content)
    #     # Unzip
    #     with zipfile.ZipFile(population_file, 'r') as zip_ref:
    #         zip_ref.extractall(population_directory)

    pop = pd.read_excel(
        context.projectpath() + '/Data/Original/Population/Elenco-codici-statistici-e-denominazioni-delle-unita-territoriali/Elenco-codici-statistici-e-denominazioni-al-01_01_2020.xls')
    pop = pop.loc[pop['Denominazione regione'] == 'Lombardia']
    pop = pop[
        ['Denominazione in italiano', 'Popolazione legale 2011 (09/10/2011)', 'Codice Comune formato alfanumerico']]
    pop = pop.merge(comuni, how='outer', left_on='Codice Comune formato alfanumerico', right_on='PRO_COM',
                    indicator='m', validate='1:1')
    assert pop['m'].unique() == 'both'
    pop.drop(columns=['PRO_COM', 'm'], inplace=True)
    pop.columns = ['comune', 'pop', 'istat', 'km2']
    pop['km2'] = pop['km2'] / 1000000  # From m2 to km2

    # Replace name of municipalities that have changed name (new to old name)
    replace_dict = {'Piadena Drizzona': 'Piadena', 'Cornale e Bastida': 'Cornale',
                    'Borgo Mantovano': 'Pieve di Coriano',
                    'Borgocarbonara': 'Borgofranco sul Po'}  # 'Sermide e Felonica': 'Sermide',
    pop['comune'].replace(replace_dict, inplace=True)
    return pop


def secondary_pm(no3m_ug, so42m_ug, nh4p_ug):
    """
    >>> no3m_ug = 9.45757954925166
    >>> so42m_ug = 4.44112367321518
    >>> nh4p_ug = 5.06366351675682
    >>> secondary_pm(no3m_ug, so42m_ug, nh4p_ug)
    (12.203328450647303, 6.759038288576356)

    :param no3m_ug:
    :param so42m_ug:
    :param nh4p_ug:
    :return:
    """
    import numpy as np

    pesomol_no3m = 62
    pesomol_so42m = 48
    pesomol_nh4p = 18

    no3m_nmoleq = no3m_ug / pesomol_no3m * 1000
    so42m_nmoleq = so42m_ug / pesomol_so42m * 1000
    nh4p_nmoleq = nh4p_ug / pesomol_nh4p * 1000

    nh4h_mnoleq = no3m_nmoleq
    nh4h_ug = nh4h_mnoleq * 18 / 1000
    if isinstance(nh4p_nmoleq, np.ndarray):
        nh4x_nmoleq = np.where(nh4p_nmoleq - nh4h_mnoleq > 0, nh4p_nmoleq - nh4h_mnoleq, 0)
    else:
        nh4x_nmoleq = nh4p_nmoleq - nh4h_mnoleq if nh4p_nmoleq - nh4h_mnoleq > 0 else 0
    nh4x_ug = nh4x_nmoleq * 18 / 1000

    nh4_no3 = nh4h_ug + no3m_ug
    nh4_2_so2 = nh4x_ug + so42m_ug

    return nh4_no3, nh4_2_so2
