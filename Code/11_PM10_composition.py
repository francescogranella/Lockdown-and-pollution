import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import context
from utils import secondary_pm

context.pdsettings()


def prep(df):
    df = df[list(df.columns[3:4]) + ['PM', 'NO3-', 'SO42-', 'NH4+']]
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df = df.drop(0)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['PM', 'NO3-', 'SO42-', 'NH4+']:
        df.loc[df[col].astype(str).str.contains('---'), col] = np.nan
        df.loc[df[col].astype(str).str.contains('<'), col] = 0
        df[col] = pd.to_numeric(df[col])
    return df


df = pd.read_excel(
    r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\Covid\Data\Original\Pollution\MI_composizione_2020+Schivenoglia.xlsx",
    sheet_name='MI_Pascal PM10_IC', header=7)
df = prep(df)

df1719 = pd.read_excel(
    r"C:\Users\Granella\Dropbox (CMCC)\PhD\Research\Covid\Data\Original\Pollution\MI_composizione_2017_2019.xlsx",
    sheet_name='MI_Pascal PM10_IC', header=7)
df1719 = prep(df1719)
mask = ((df1719['date'] > pd.to_datetime('2017-02-21')) & (df1719['date'] < pd.to_datetime('2017-05-05'))) | \
       (df1719['date'] > pd.to_datetime('2018-02-21')) & (df1719['date'] < pd.to_datetime('2018-05-05')) | \
       (df1719['date'] > pd.to_datetime('2019-02-21')) & (df1719['date'] < pd.to_datetime('2019-05-05'))
df1719 = df1719[mask]

df = pd.concat([df1719, df])
df['2020'] = (df['date'].dt.year == 2020) * 1

df = df.loc[df['date'] < pd.to_datetime('2020-05-05')]
ammonium_nitrates, ammonium_sulfates = secondary_pm(df['NO3-'].values, df['SO42-'].values, df['NH4+'].values)
df['an'], df['as'] = ammonium_nitrates, ammonium_sulfates
df['ratio_n2pm'] = (df['an'] + df['as']) / df['PM']

df['year'] = df['date'].dt.year
df['date'] = pd.to_datetime(dict(year=2020, month=df.date.dt.month, day=df.date.dt.day))

df.to_stata(context.projectpath() + '/Data/Modified/composition.dta', write_index=False)

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


fig, ax = plt.subplots(figsize=(12, 4))
fontsize = 12
ax.plot(df['date'], df['ratio_n2pm'], label='[(NH4)NO3 + (NH4)2SO2]/PM')
ax.set_ylim((0, 1))
format_ticks(ax)
vertical_lines(ax)
rotate_ticks()
place_text(ax, fontsize=fontsize)
ax.set_title('Ammonium sulfate and ammonium nitrate as a share of PM$_{10}$', fontsize=fontsize)
plt.tight_layout()
plt.savefig(context.projectpath() + '/Docs/img/nitrates_to_pm.eps')
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
fontsize = 12
ax.plot(df.loc[df['year'] == 2017, 'date'], df.loc[df['year'] == 2017, 'ratio_n2pm'], label='2017')
ax.plot(df.loc[df['year'] == 2018, 'date'], df.loc[df['year'] == 2018, 'ratio_n2pm'], label='2018')
ax.plot(df.loc[df['year'] == 2019, 'date'], df.loc[df['year'] == 2019, 'ratio_n2pm'], label='2019')
ax.plot(df.loc[(df['year'] == 2020) & (df.date > pd.to_datetime('2020-02-21')), 'date'],
        df.loc[(df['year'] == 2020) & (df.date > pd.to_datetime('2020-02-21')), 'ratio_n2pm'], label='2020')
ax.set_ylim((0, 1))
format_ticks(ax)
vertical_lines(ax)
rotate_ticks()
place_text(ax, fontsize=fontsize)
ax.set_title('Ammonium sulfate and ammonium nitrate as a share of PM$_{10}$', fontsize=fontsize)
plt.legend()
plt.tight_layout()
# plt.savefig(context.projectpath() + '/Docs/img/nitrates_to_pm.png')
plt.show()
