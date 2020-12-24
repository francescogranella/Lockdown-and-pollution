"""
XGB SCRATCH
"""
import logging
import pickle
import traceback
from datetime import datetime

import context
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from shapely.geometry import Point
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from tabulate import tabulate
from tqdm import tqdm
from xgboost import XGBRegressor

context.pdsettings()

if __name__ == '__main__':
    logging.basicConfig(filename=context.projectpath() + '/XGB.log', format='\n%(asctime)s \n%(message)s\n',
                        level=logging.INFO, filemode='w')

start_time = datetime.now()
print('start', start_time)

inputfile = context.projectpath() + '/Data/Modified/Pollution+Meteo.parq'
data = pd.read_parquet(inputfile)

dropcols = [x for x in data.columns if 'temp_min' in x or 'temp_max' in x]
data.drop(columns=dropcols, inplace=True)

start = '2020-01-02'
end = '2020-05-03'


class IsolateTarget(BaseEstimator):
    """
    Drop non-target pollution stations. Drop ammonia if needed. Purge missing values in dependent var
    """

    def __init__(self, idsensore):
        self.idsensore = idsensore

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Drop non-target pollution stations
        drop_cols = [x for x in X.columns if ('NO2' in x or 'PM2.5' in x) and str(self.idsensore) not in x]
        X.drop(columns=drop_cols, inplace=True)
        target_col = [x for x in X.columns if str(self.idsensore) in x]
        assert len([x for x in X.columns if str(self.idsensore) in x]) == 1
        target_col = target_col[0]
        X.rename(columns={target_col: 'y'}, inplace=True)
        # Purge missing values in dependent var
        X = X.loc[~pd.isna(X['y'])]
        return X


class MonthYear(BaseEstimator):
    """
    Keep only choseon months, year
    """

    def __init__(self, months=None, first_year=2015):
        if months is None:
            months = [10, 11, 12, 1, 2, 3, 4]
        self.months = months
        self.first_year = first_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Only recent years
        X = X[X['year'] >= self.first_year]
        # Only relevant months]
        X = X[X['dt'].dt.month.isin(self.months)]
        return X


class WithinRadius(BaseEstimator):
    """
    Keep only weather stations within radius of target pollution monitor
    """

    def __init__(self, idsensore, radius=0.75):
        self.radius = radius
        self.idsensore = idsensore

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Coordinates of target monitor
        pollution_metadata = pd.read_parquet(context.projectpath() + '/Data/Modified/PollutionStations.parq')
        pollution_metadata = pollution_metadata[['idsensore', 'lng', 'lat']]
        lng, lat = pollution_metadata.loc[pollution_metadata['idsensore'] == self.idsensore, ['lng', 'lat']].iloc[
            0].to_list()

        # Identify weather stations within radius of target pollution station
        weather_metadata = pd.read_parquet(context.projectpath() + '/Data/Modified/MeteoStations.parq')
        gdf = gpd.GeoDataFrame(weather_metadata['idsensore'],
                               geometry=gpd.points_from_xy(weather_metadata['lng'], weather_metadata['lat']),
                               crs='EPSG:4326')
        gdf['dist'] = gdf.geometry.distance(Point(lng, lat))

        # Drop distant weather stations
        gdf = gdf[gdf['dist'] > self.radius]
        distant_stations = gdf['idsensore'].to_list()

        # Only target station and meteo stations within radius
        distant_stations_cols = [x for s in distant_stations for x in X.columns if str(s) in x]
        X.drop(columns=distant_stations_cols, inplace=True)
        return X


class WithinLag(BaseEstimator):
    """Choose how far back features are lagged"""

    def __init__(self, max_lag=7):
        self.max_lag = max_lag

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lag_cols = [x for x in X.columns for lag in range(self.max_lag + 1, 8) if f'_l{lag}' in x]
        X.drop(columns=lag_cols, inplace=True)
        return X


class DropExcessiveMissing(BaseEstimator):
    """Drop feature columns with too many missing values"""

    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X.replace({-9999: np.nan}, inplace=True)
        na_sum = X.isna().sum() / len(X)
        na_cols = na_sum[na_sum > self.threshold].index
        X.drop(columns=na_cols, inplace=True)
        return X


class DropConstantCols(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.loc[:, (X != X.iloc[0]).any()]
        return X


class FillNa(BaseEstimator):
    "Fill missing values with -9999"

    def __init__(self, fill=True):
        self.fill = fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.fill:
            X.fillna(-9999, inplace=True)
        return X


class DropDate(BaseEstimator):
    """Self-explanatory"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.drop(columns=['dt'], inplace=True)
        return X


class ImputeNA(BaseEstimator):
    def __init__(self, impute=False):
        self.impute = impute

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.impute:
            impute_cols = list(set(X.columns) - set(['dt', 'y']))
            I = X[impute_cols]
            I = (I - I.min()) / (I.max() - I.min())
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=10, copy=True)
            I = pd.DataFrame(imputer.fit_transform(I), columns=impute_cols)
            X[impute_cols] = I
        return X


class CustomOLS(BaseEstimator):
    """OLS prediction with selected features"""

    def __init__(self, cols):
        self.cols = cols
        pass

    def fit(self, X, y=None):
        xtrain = pd.DataFrame(X, columns=self.cols)
        xtrain = xtrain[['Lifted_index12', 'sin5', 'year']]
        xtrain.replace({-9999: np.nan}, inplace=True)
        xtrain.fillna(xtrain.median(), inplace=True)
        # xtrain['sin5year'] = xtrain['sin5']*xtrain['year']
        self.ols = LinearRegression()
        self.ols.fit(xtrain, y)
        return self

    def predict(self, X, y=None):
        xtrain = pd.DataFrame(X, columns=self.cols)
        xtrain = xtrain[['Lifted_index12', 'sin5', 'year']]
        xtrain.replace({-9999: np.nan}, inplace=True)
        xtrain.fillna(xtrain.median(), inplace=True)
        # xtrain['sin5year'] = xtrain['sin5']*xtrain['year']
        yhat = self.ols.predict(xtrain)
        return yhat


def custom_cv_func(xtrain):
    """
    Prepares custom cross-validation folds
    :param xtrain:
    :return:
    """
    fold = []
    for year in [2016, 2017, 2018, 2019]:
        ix_test_cs = xtrain[(xtrain['year'] == year) & (xtrain['month'].isin([1, 2, 3, 4]))].index
        ix_train_cv = list(set(xtrain.index) - set(ix_test_cs))
        fold.append((ix_train_cv, ix_test_cs))
    return fold


def test_train(df, start, end):
    """
    Split data in train and test
    """
    # Train-test split
    target_period_start = pd.to_datetime(start, format='%Y-%m-%d')
    target_period_end = pd.to_datetime(end, format='%Y-%m-%d')

    df.drop(columns=list(set(df.columns) & set(x for x in data.columns if 'day' in x or 'week' in x)), inplace=True)

    train = df.loc[(df['dt'] < target_period_start) | (df['dt'] > target_period_end)]
    test = df.loc[(df['dt'] >= target_period_start) & (df['dt'] <= target_period_end)]

    xtrain = train[[x for x in train.columns if x != 'y' and x != 'dt']].reset_index(drop=True)
    ytrain = train['y'].reset_index(drop=True)
    xtest = test[[x for x in test.columns if x != 'y' and x != 'dt']]
    ytest = test['y']

    dates_train = train['dt']
    dates_test = test['dt']

    return xtrain, ytrain, xtest, ytest, dates_train, dates_test


idsensori = [int(x.split('_')[0]) for x in data.columns if 'NO2' in x]
for idsensore in idsensori:
    data_param_grid = [
        {'idsensore': [idsensore],
         'months': [[1, 2, 3, 4, 5, 10, 11, 12]],
         'first_year': [2012],
         'radius': [0.8],
         'max_lag': [2],
         }
    ]

    # Catch bad stations
    bad_stations = [20358, 20443, 17162]
    if idsensore in bad_stations:
        continue

    # Model parameter space for grid search
    model_param_grid = {'learning_rate': np.linspace(0.001, 0.15, num=100),
                        'subsample': np.linspace(0.45, 1.0, num=100),
                        'n_estimators': np.arange(100, 1500, step=10),
                        'max_features': [None, 'sqrt'],
                        'objective': ['reg:squarederror']
                        }

    logging.info('\n Model parameter grid:\n' + tabulate(model_param_grid, headers='keys') + '\n')
    logging.info('\n Data parameter grid:\n' + tabulate(data_param_grid, headers='keys') + '\n')

    # Search the data pre-processing that yields the best results

    # Compbinations of pre-processing parameters
    combinations = [list(ParameterGrid(x)) for x in data_param_grid]
    combinations = [x for y in combinations for x in y]

    # Catch outputs
    model_list = []
    prediction_list = []
    comb_metric_list = []
    fi_list = []
    # Iterate over combinations of pre-processing parameters
    try:
        for comb in tqdm(combinations):

            print('\n' + tabulate({k: [v] for k, v in comb.items()}, headers='keys'))

            # Pre-process data
            pipe = make_pipeline(IsolateTarget(idsensore=comb['idsensore']),
                                 MonthYear(months=comb['months'], first_year=comb['first_year']),
                                 WithinRadius(idsensore=comb['idsensore'], radius=comb['radius']),
                                 WithinLag(comb['max_lag']),
                                 DropConstantCols(),
                                 # DropExcessiveMissing(comb['na_threshold']),
                                 # ImputeNA(impute=comb['impute']),
                                 # FillNa(fill=comb['z_fill'])
                                 )

            df = pipe.transform(data.copy())

            # Train-test split.
            xtrain, ytrain, xtest, ytest, dates_train, dates_test = test_train(df, start, end)

            # Evaluation set. Not used
            eval_set = pd.concat([dates_test, xtest, ytest], axis=1)
            eval_set = eval_set.loc[eval_set['dt'] < pd.to_datetime('20200223')]
            eval_set_x = eval_set[[x for x in eval_set.columns if x != 'dt' and x != 'y']]
            eval_set_y = eval_set['y']

            # Custom cross-validation
            custom_cv = custom_cv_func(xtrain)

            n_iter = 50
            n_jobs_rscv = 50
            n_jobs_xgb = 1

            print('Now to XGB')
            print(f'# iter {n_iter}, njobs search {n_jobs_rscv}, n_jobs XGB {n_jobs_xgb}')

            # Estimator
            estimator = XGBRegressor(seed=1233, verbosity=1, n_jobs=n_jobs_xgb)
            model = RandomizedSearchCV(estimator=estimator, param_distributions=model_param_grid, cv=custom_cv,
                                       verbose=1,
                                       n_jobs=n_jobs_rscv, scoring='neg_root_mean_squared_error', n_iter=n_iter,
                                       random_state=1233)

            try:
                # Cross-validated randomized paramater search
                model.fit(xtrain, ytrain)
                logging.info('\n' + tabulate({k: [v] for k, v in comb.items()}, headers='keys'))
            # Catch with parallelization problems on server
            except PermissionError:
                print('fail')
                model.n_jobs = 1
                model.fit(xtrain, ytrain)
                logging.info(f"{comb}")
                logging.info(tabulate({k: [v] for k, v in comb.items()}, headers='keys'))
            except ValueError:
                logging.error(f"{comb['idsensore']} FAILED")
                print(f"{comb['idsensore']} FAILED")
                fail_token = True
                break

            fail_token = False


            # Evalutaion metrics
            def metrics(y_true, y_pred):
                r2 = r2_score(y_true, y_pred)
                mean_bias = np.mean(y_true - y_pred)
                nmean_bias = mean_bias / np.mean(y_true)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                crmse = mean_squared_error(y_true, y_pred - np.mean(y_pred) + np.mean(y_true), squared=False)
                ncrmse = crmse / np.mean(y_true)
                pcc, _ = pearsonr(y_true, y_pred)
                return dict(r2=r2, mean_bias=mean_bias, nmean_bias=nmean_bias, rmse=rmse, crmse=crmse, ncrmse=ncrmse,
                            pcc=pcc)


            # In-sample prediction
            train_pred = model.predict(xtrain)
            train_metrics = pd.DataFrame(metrics(ytrain, train_pred), index=[0])
            train_metrics.columns = ['train_' + x for x in train_metrics.columns]

            # Out-of-sample prediction
            test_pred = model.predict(xtest)
            test_metrics = pd.DataFrame(metrics(ytest, test_pred), index=[0])
            test_metrics.columns = ['test_' + x for x in test_metrics.columns]

            # Model parameters
            model_params = pd.DataFrame(model.best_params_, index=[0])

            # Store monitor id, pre-processing parameters, model parameters, evaluation metrics
            comb_df = pd.DataFrame({k: [v] for k, v in comb.items()})
            comb_metrics = pd.concat([comb_df, model_params, test_metrics, train_metrics], axis=1)

            print('\n' + tabulate(comb_metrics, headers='keys'))
            logging.info('\n' + tabulate(comb_metrics, headers='keys'))

            # Store model predictions
            prediction = pd.DataFrame(
                dict(idsensore=comb['idsensore'], dt=dates_test.values, Observed=ytest.values, Predicted=test_pred))
            prediction = pd.merge(prediction, comb_metrics, on='idsensore')
            prediction_list.append(prediction)

            # Store model
            model.idsensore = comb['idsensore']
            model_list.append(model)

            # Store feature importances
            fi = pd.DataFrame(dict(
                zip(model.best_estimator_.get_booster().feature_names, model.best_estimator_.feature_importances_)),
                              index=[0]).T.reset_index()
            fi.columns = ['feature', 'importance']
            fi['idsensore'] = comb['idsensore']
            fi_list.append(fi)

        date_time = datetime.now().strftime("%m-%d-%Y %Hh%M")

        # Save model predictions
        predictions_df = pd.concat(prediction_list, axis=0)
        predictions_df.to_parquet(context.projectpath() + f'/Data/Modified/Predictions/Predictions {date_time}.parq')
        predictions_df['months'] = predictions_df['months'].transform(lambda x: str(x))
        predictions_df.replace({None: 'None'}, inplace=True)
        predictions_df.to_stata(context.projectpath() + f'/Data/Modified/Predictions/Predictions {date_time}.dta',
                                write_index=False)

        # Save models
        with open(context.projectpath() + f'/Models/Model {date_time}.pkl', 'wb') as f:
            pickle.dump(model_list, f, protocol=4)

        # Save feature importances
        pd.concat(fi_list, axis=0).to_parquet(context.projectpath() + f'/Models/FeatureImportances {date_time}.parq')

        all_well = True

    except Exception as e:
        logging.error(traceback.format_exc())

        all_well = False

    # Timing
    end_time = datetime.now()
    print('end', end_time)
    print('start at', start_time, 'end at', end_time)
    total_time = end_time - start_time
    print('total time', end_time - start_time)
