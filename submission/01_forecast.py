import pandas as pd
import plotly.express as px
from darts.models import AutoARIMA
from darts import TimeSeries
import matplotlib.pyplot as plt
from darts.models import NaiveDrift, ExponentialSmoothing, Prophet 
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from sklearn.metrics import get_scorer_names, mean_absolute_percentage_error, make_scorer

warnings.simplefilter(action='ignore', category=Warning)

data = pd.read_csv("../data/Datafest2024_Train.csv")
test = pd.read_csv("../data/Datafest2024_Test.csv")

#test['fecha_transaccion'].min()
#data = data.append(test)

#data.head()
data["fecha_transaccion"] = pd.to_datetime(data["fecha_transaccion"], format="%Y%m%d")

def create_features(df, label=None):
#    df['date'] = pd.to_datetime(df['fecha_transaccion'])
    df['year'] = df['fecha_transaccion'].dt.year
    df['month'] = df['fecha_transaccion'].dt.month
    df['day'] = df['fecha_transaccion'].dt.day
    df['dayofweek'] = df['fecha_transaccion'].dt.dayofweek
    df['quarter'] = df['fecha_transaccion'].dt.quarter

    
    X = df[['date','year', 'month', 'day', 'dayofweek', 'quarter']]
    if label:
        y = df[label]
        return X, y
    return X

df = data[data['codigo_cajero'] == 150]
df['demanda'] = np.where(df['demanda'] < 0, np.nan, df['demanda'])
df['demanda'] = df['demanda'].fillna(value=df['demanda'].mean())

data = data[data["fecha_transaccion"]<'2024-05-21']
X, y = create_features(data, lbel='demanda')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LGBMRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    num_leaves=31
)
model.fit(X_train, y_train)

from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

X, y = create_features(df, label='demanda')
X = X.sort_values(by=['date'])
plt.plot(X['date'],y)

splits = 10
tscv = TimeSeriesSplit(n_splits=splits, test_size=7)
accumulated_mape = 0

fig = plt.figure(figsize=(20, 5))
plt.plot(X['date'][30:], y[30:])
for train, test in tscv.split(X):

    x_= X.drop(columns=["date"])
    model.fit(x_.iloc[train], y.iloc[train])
    y_pred = model.predict(x_.iloc[test])

    #plt.plot(X.iloc[train]['date'], y.iloc[train])
    plt.plot(X.iloc[test]['date'], y_pred)
    #plt.plot(X.iloc[test]['date'], y.iloc[test])
    curr_mape = mean_absolute_percentage_error(y.iloc[test], y_pred)
    print(curr_mape)
    accumulated_mape += curr_mape

plt.show()

accumulated_mape
print(f"mean mape: {accumulated_mape/splits}")


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


scorer = make_scorer(mean_absolute_percentage_error)
get_scorer_names()
cross_val_score(model, X, y, cv=10, scoring=scorer)
scores = cross_val_score(model, X, y, cv=7, scoring=scorer)
int(scores[0])
print(scores.mean())

rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

code_atms = data["codigo_cajero"].unique().tolist()
code_atms.sort()

X, y = create_features(data, label='demanda')
y.head()
X.head()
result = pd.DataFrame()

for atm in code_atms:
    single_atm = data[data["codigo_cajero"] == atm] 
    single_atm = single_atm.sort_values(by=['fecha_transaccion'])
    single_atm = single_atm.append({
        'fecha_transaccion': pd.to_datetime('2024-05-27'),
        'codigo_cajero': atm,
        'tipo_cajero': single_atm['tipo_cajero'].iloc[0],
        'saldo_inicial': np.nan,
        'demanda': np.nan,
        'abastecimiento': np.nan,
        'saldo_final': np.nan,
     }, ignore_index=True)

    single_atm['demanda'] = np.where(single_atm['demanda'] < 0, np.nan, single_atm['demanda'])
    single_atm['demanda'] = single_atm['demanda'].fillna(value=single_atm['demanda'].mean())
    X, y = create_features(single_atm, label='demanda')
    #if atm == 1:
    #    break

    X_train = X[:-7]
    y_train = y[:-7]
    X_pred = X[-7:]
    #single_atm_pred = single_atm[single_atm["fecha_transaccion"]<'2024-05-21']

    model = LGBMRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42,
        num_leaves=31
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)
    #ts = TimeSeries.from_dataframe(
    #    single_atm_pred,
    #    time_col='fecha_transaccion',
    #    value_cols=['demanda']
    #)
    #model = AutoARIMA(start_p=8, max_p=12, start_q=1, n_jobs=-1)
    #model = ExponentialSmoothing()
    #model.fit(ts)
    #pred = model.predict(7)

    #pred = pred.pd_dataframe().reset_index()
    history = single_atm["demanda"].copy().values[:-7].tolist()
    #history.extend(pred['demanda'].values)
    history.extend(y_pred.tolist())
    single_atm["demanda"] = history
    single_atm = single_atm[single_atm['fecha_transaccion'] >'2024-04-14'] 
    single_atm = single_atm[['fecha_transaccion', 'codigo_cajero', 'tipo_cajero', 'saldo_inicial', 'demanda', 'abastecimiento', 'saldo_final']]
    result = result.append(single_atm)
    print(atm)

result
import matplotlib.pyplot as plt 
v4 = pd.read_csv('../output/naive_results_v4.csv')
v4['fecha_transaccion'].min()
v4[v4['codigo_cajero'] == 1].plot(x='fecha_transaccion', y='demanda', figsize=(20,5))
result[result['codigo_cajero'] == 1].plot(x='fecha_transaccion', y='demanda', figsize=(20,5))
plt.show()

result_ = result[result['abastecimiento'].isna()]
result_

result_['fecha_transaccion'] = result_['fecha_transaccion'].astype(str)
result_
df_output = (
    result_.pivot(index='codigo_cajero', columns='fecha_transaccion', values='demanda')
    .reset_index().reset_index(drop=True)
)

df_output = (
    df_output.merge(
        right=data[['codigo_cajero', 'tipo_cajero']].drop_duplicates(keep='first'),
        on='codigo_cajero',
        how='left'
    )
)
df_output['fecha_transaccion'] = '2024-05-20'
date_cols = ['2024-05-21', '2024-05-22', '2024-05-23', '2024-05-24', '2024-05-25', '2024-05-26', '2024-05-27']
main_cols = ['fecha_transaccion', 'codigo_cajero', 'tipo_cajero']

df_output_ = df_output[main_cols + date_cols]
df_output_
new_date_cols = [f"demanda_{d.replace('-', '')}" for d in date_cols]
df_output_.columns = main_cols + new_date_cols

df_output_.to_csv('../output/forecast_v1.csv')
result.to_csv("../output/naive_results_v5.csv")
result['fecha_transaccion'].min()