import glob
import io
import os
import random
import time
from datetime import datetime
from io import StringIO
from zipfile import ZipFile

import context
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

context.pdsettings()


def solve_mixed_types(df):
    for col in df.columns:
        # Find obervations whose type is different from the column's type as inferred by pandas
        weird = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis=1)
        if len(df[weird]) > 0:
            # If there are any weird observations in the column, convert the column to string
            print(col)
            df[col] = df[col].astype(str)
        # If the column has >1 type, convert the column to string
        if df[col].dtype == list:
            # Otherwise, keep
            df[col] = df[col].astype(str)
        else:
            pass
    return df


def url_to_parquet(year, url, overwrite=False):
    # Check if file doesn't already exist
    there = os.path.isfile(context.projectpath() + f'/Data/Original/Weather/Meteo{year}.parq')
    if there and not overwrite:
        print(year, 'already there')
        return
    else:
        print('Reading from URL')
        df = pd.read_csv(url)
        print('Solving mixed types')
        df = solve_mixed_types(df)
        # Save original file
        print('Saving')
        df.to_parquet(context.projectpath() + f'/Data/Original/Weather/Meteo{year}.parq')


# 1 - My way: own script
class LinateAtmoSoundingData:

    def __init__(self, from_date, to_date, wait=0):
        self.from_date = from_date
        self.to_date = to_date
        self.wait = wait
        self.stid = 'LIML'
        self.col_dict = {'PRES': 'pressure', 'HGHT': 'height', 'TEMP': 'temperature', 'DWPT': 'dewpoint',
                         'FRPT': 'frostpoint', 'RELH': 'relhumidity', 'RELI': 'relhumidityice', 'MIXR': 'mixingratio',
                         'DRCT': 'direction', 'SKNT': 'speed', 'THTA': 'ptemp', 'THTE': 'eqptemp',
                         'THTV': 'virtualptemp'}

    def run(self):

        dflist = []
        daterange = pd.date_range(self.from_date, self.to_date, freq='M', closed='right')
        progbar = tqdm(daterange, total=len(daterange) + 1)

        for i, date in enumerate(progbar):
            from_date = datetime(date.year, date.month, 1, 0)
            to_date = datetime(date.year, date.month, date.day, 12)

            progbar.set_description(f"Processing {from_date.year}-{from_date.month}")

            temp = self._get_raw_data(from_date, to_date)
            dflist.append(temp)

            if i > 0 and i % 10 == 0:
                df = pd.concat(dflist, sort=False)
                for col in df.columns:
                    if col not in ['Station identifier', 'date']:
                        df[col] = df[col].astype(float)
                df.to_stata(context.projectpath() + f"/Data/Modified/AtmoSounding_.dta", write_index=False)

            time.sleep(self.wait)

        today = datetime.today()
        if daterange[-1] < today:
            from_date = datetime(today.year, today.month, 1, 0)
            to_date = datetime(today.year, today.month, today.day - 1, 12)
            temp = self._get_raw_data(from_date, to_date)
            dflist.append(temp)

        return pd.concat(dflist, sort=False)

    def _get_raw_data(self, from_date, to_date):

        self.raw_data = requests.get(
            f'http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ALIST&YEAR={from_date:%Y}&MONTH={from_date:%m}&FROM={from_date:%d%H}&TO={to_date:%d%H}&STNM={self.stid}').text

        while 'Connection Timed Out' in self.raw_data or 'Sorry, the server is too busy' in self.raw_data:
            print('\nWaiting..', from_date, '\n', self.raw_data.split('\n')[:50])
            time.sleep(random.randint(90, 120))
            self.raw_data = requests.get(
                f'http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ALIST&YEAR={from_date:%Y}&MONTH={from_date:%m}&FROM={from_date:%d%H}&TO={to_date:%d%H}&STNM={self.stid}').text

        self.soup = BeautifulSoup(self.raw_data, 'html.parser')

        n_iterations = len(self.soup.find_all('pre'))

        dflist = []
        for measurement in range(0, n_iterations, 2):
            dflist.append(self._parse(measurement))

        return pd.concat(dflist, sort=False)

    def _parse(self, i):

        if i > 0 & i % 2 == 1:
            raise ValueError(f'{i} is not even')

        col_names = self.soup.find_all('pre')[i].contents[0].split('\n')[2].split('   ')
        col_names.remove('')
        col_names = [self.col_dict[x] for x in col_names]
        tabular_data = StringIO(self.soup.find_all('pre')[i].contents[0])
        vertical_profile = pd.read_fwf(tabular_data, skiprows=5, names=col_names)

        tabular_data = StringIO(self.soup.find_all('pre')[i + 1].contents[0])
        indices = pd.read_fwf(tabular_data, skiprows=1, names=[0, 1]).set_index(0).T
        indices.columns = [x.replace(':', '') for x in indices.columns]
        date = indices['Observation time'].values[0]
        date = pd.to_datetime(date, format='%y%m%d/%H00')
        indices['date'] = date
        indices.drop(columns='Observation time', inplace=True)

        return pd.concat([vertical_profile, pd.concat([indices] * len(vertical_profile), ignore_index=True)], axis=1)


########################################################################################################################
# POLLUTION DATA

# Download
pollution_urls_dict = {
    2020: "https://www.dati.lombardia.it/api/views/nicp-bhqi/rows.csv?accessType=DOWNLOAD",
    2019: "https://www.dati.lombardia.it/api/views/kujm-kavy/rows.csv?accessType=DOWNLOAD",
    2018: "https://www.dati.lombardia.it/api/views/bgqm-yq56/rows.csv?accessType=DOWNLOAD",
    2017: "https://www.dati.lombardia.it/api/views/fdv6-2rbs/files/742fb7a8-2a58-4b08-a366-a75c358be1ed?filename=sensori_aria_2017.zip",
    2016: "https://www.dati.lombardia.it/api/views/7v3n-37f3/files/cef6571a-d4e3-42c4-8a44-9a504c77ef9c?filename=sensori_aria_2016.zip",
    2015: "https://www.dati.lombardia.it/api/views/bpin-c7k8/files/dd35ff6d-2645-404e-bcd4-0e49d4af3fc2?filename=sensori_aria_2015.zip",
    2014: "https://www.dati.lombardia.it/api/views/69yc-isbh/files/487864fb-1f8c-4a28-9949-c4d64adacd29?filename=sensori_aria_2014.zip",
    2013: "https://www.dati.lombardia.it/api/views/hsdm-3yhd/files/ce297c9e-59bd-48f3-b4fb-f1baf7e93840?filename=sensori_aria_2013.zip",
    2012: "https://www.dati.lombardia.it/api/views/wr4y-c9ti/files/72c5105a-e1f3-46fe-873b-f3dd449fd6ca?filename=sensori_aria_2012.zip",
    2011: "https://www.dati.lombardia.it/api/views/5mut-i45n/files/6cdf8360-beea-42e7-b9d4-774e06147d6c?filename=sensori_aria_2011.zip",
    2010: "https://www.dati.lombardia.it/api/views/wp2f-5nw6/files/e780233e-18fc-4e46-8417-7bb2af435f73?filename=sensori_aria_2008-2010.zip",
}

# Download zip (without saving) and extract CSVs
for year, url in pollution_urls_dict.items():
    print(year)
    req = requests.get(url)
    if year <= 2017:
        filebytes = io.BytesIO(req.content)
        myzipfile = ZipFile(filebytes)
        myzipfile.extractall(context.projectpath() + '/Data/Original/Pollution/')
    elif year > 2017:
        with open(context.projectpath() + f'/Data/Original/Pollution/{year}.csv', 'wb') as f:
            f.write(req.content)


def pollution_modify_save(year, df):
    # Lowercase
    df.columns = [x.lower() for x in df.columns]
    # Datetime format
    if year > 2017:
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %I:%M:%S %p')
    else:
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M:%S')
    # Valid observations
    df['valid'] = (df['stato'] == 'VA') * 1
    df.drop(columns='stato', inplace=True)
    # Solve mixed column tyoes
    df = solve_mixed_types(df)
    # Save
    df.to_parquet(context.projectpath() + f'/Data/Modified/Pollution{year}.parq')


# Metadata
pollution_metadata = pd.read_csv("https://www.dati.lombardia.it/api/views/ib47-atvt/rows.csv?accessType=DOWNLOAD")
# Lowercase
pollution_metadata.columns = [x.lower() for x in pollution_metadata.columns]
# Datetime format
pollution_metadata['datastart'] = pd.to_datetime(pollution_metadata['datastart'], format='%d/%m/%Y')
pollution_metadata['datastop'] = pd.to_datetime(pollution_metadata['datastop'], format='%d/%m/%Y')
# Rename
pollution_metadata.rename(columns={'unitamisura': 'unit', 'nometiposensore': 'pollutant'}, inplace=True)
# Pollutant short name
shorten_dict = {'Ozono': 'O3', 'Monossido di Carbonio': 'CO', 'Nikel': 'Ni', 'Ossidi di Azoto': 'NOx', 'PM10': 'PM10',
                'Biossido di Azoto': 'NO2', 'Particolato Totale Sospeso': 'PM', 'Biossido di Zolfo': 'SO2',
                'Particelle sospese PM2.5': 'PM2.5', 'BlackCarbon': 'BlackCarbon', 'Benzene': 'C6H6',
                'Ammoniaca': 'NH3', 'Arsenico': 'As', 'Monossido di Azoto': 'NO', 'Cadmio': 'Cd', 'Piombo': 'Pb',
                'Benzo(a)pirene': 'Benzo[a]pyrene'}

pollution_metadata['pollutantshort'] = pollution_metadata['pollutant'].replace(shorten_dict)
# Save
pollution_metadata.to_parquet(context.projectpath() + '/Data/Modified/PollutionStations.parq')

# Actual data
files = glob.glob(context.projectpath() + '/Data/Original/Pollution/20*.csv')
for file in files:
    year = int(file[-8:-4])
    print(year)
    # if os.path.isfile(context.projectpath() + f'/Data/Modified/Pollution{year}.parq'):
    #     continue
    df = pd.read_csv(file)
    pollution_modify_save(year, df)

########################################################################################################################
# WEATHER DATA

# Metadata
weather_meta = pd.read_csv("https://www.dati.lombardia.it/api/views/nf78-nj6b/rows.csv?accessType=DOWNLOAD")
weather_meta.metadata = """idOperatore: 	1 = Average value
				3 = Maximum value
				4 = Cumulative value
				2 = ?"""

# Lowercase
weather_meta.columns = [x.lower() for x in weather_meta.columns]
# Datetime format
weather_meta['datastart'] = pd.to_datetime(weather_meta['datastart'], format='%d/%m/%Y')
weather_meta['datastop'] = pd.to_datetime(weather_meta['datastop'], format='%d/%m/%Y')
# Rename column
weather_meta.rename({'unità dimisura': 'unit'}, inplace=True)
# Transalte type of var
typesdict = {'Temperatura': 'temp', 'Direzione Vento': 'winddir', 'Velocità Vento': 'windspeed',
             'Precipitazione': 'prec', 'Umidità Relativa': 'hum'}
weather_meta['tipologia'].replace(typesdict, inplace=True)
weather_meta = weather_meta[weather_meta['tipologia'].isin(['winddir', 'prec', 'temp', 'hum', 'windspeed'])]
# Save
weather_meta.to_parquet(context.projectpath() + '/Data/Modified/MeteoStations.parq')

# Download
urls_dict = {
    2020: "https://www.dati.lombardia.it/api/views/erjn-istm/files/048e675f-2133-4c43-aa8c-0297afeea79f?filename=sensori_meteo_2020.zip",
    2019: "https://www.dati.lombardia.it/api/views/wrhf-6ztd/files/242ee325-894d-4688-ae2e-6fecc08edfff?filename=sensori_meteo_2019.zip",
    2018: "https://www.dati.lombardia.it/api/views/sfbe-yqe8/files/b9830c77-e6f3-4e38-8377-c8ff7aca432a?filename=sensori_meteo_2018.zip",
    2017: "https://www.dati.lombardia.it/api/views/vx6g-atiu/files/28d621e4-1e3d-42ea-99b3-eda2586c63cd?filename=sensori_meteo_2017.zip",
    2016: "https://www.dati.lombardia.it/api/views/kgxu-frcw/files/c4cc09f0-cfae-4f39-bd91-8222317f8b80?filename=sensori_meteo_2016.zip",
    2015: "https://www.dati.lombardia.it/api/views/knr4-9ujq/files/ad28ba14-784e-4bdb-b8fa-9b49e521aa66?filename=sensori_meteo_2015.zip",
    2014: "https://www.dati.lombardia.it/api/views/fn7i-6whe/files/4dbb02f2-6997-46d4-ae57-fa8471834bd2?filename=sensori_meteo_2014.zip",
    2013: "https://www.dati.lombardia.it/api/views/76wm-spny/files/25cb666f-99f4-47d9-a03d-8e9aa4039c2c?filename=sensori_meteo_2013.zip",
    2012: "https://www.dati.lombardia.it/api/views/srpn-ykcs/files/53ae3b0d-c5f9-4cec-8e6b-9d6b7f65a750?filename=sensori_meteo_2011-2012.zip",
}

# Download zip (without saving) and extract CSVs
for year, url in urls_dict.items():
    print(year)
    req = requests.get(url)
    filebytes = io.BytesIO(req.content)
    myzipfile = ZipFile(filebytes)
    myzipfile.extractall(context.projectpath() + '/Data/Original/Weather/')


# Convert CSV to parquet

def weather_modify_save(year, df):
    # Lowercase
    df.columns = [x.lower() for x in df.columns]
    # Datetime format
    try:
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M:%S')
    except ValueError:
        df['data'] = df['data'].str.replace('.000', '')
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M:%S')
    # Valid observations
    df['valid'] = (df['stato'] == 'VA') * 1
    df.drop(columns='stato', inplace=True)
    # Solve mixed column tyoes
    df = solve_mixed_types(df)
    # Save
    df.to_parquet(context.projectpath() + f'/Data/Modified/Meteo{year}.parq')


files = glob.glob(context.projectpath() + '/Data/Original/Weather/20*.csv')
for file in files:
    year = int(file[-8:-4])
    print(year)
    if os.path.isfile(context.projectpath() + f'/Data/Modified/Meteo{year}.parq'):
        continue
    df = pd.read_csv(file)
    weather_modify_save(year, df)

########################################################################################################################
# ATMOSPHERIC SOUNDINGS
atmo = LinateAtmoSoundingData(datetime(2011, 1, 1), datetime.today(), wait=10)
df = atmo.run()
for col in df.columns:
    if col not in ['Station identifier', 'date']:
        df[col] = df[col].astype(float)
df.to_stata(context.projectpath() + "/Data/Modified/AtmoSounding.dta", write_index=False)
