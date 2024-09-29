import pandas as pd
import plotly.express as px
from darts.models import AutoARIMA
from darts import TimeSeries
import matplotlib.pyplot as plt
from darts.models import NaiveDrift
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=Warning)

data = pd.read_csv("./data/Datafest2024_Test.csv")

data.head()
data["fecha_transaccion"] = pd.to_datetime(data["fecha_transaccion"], format="%Y%m%d")

code_atms = data["codigo_cajero"].unique().tolist()
code_atms.sort()

result = pd.DataFrame()

for atm in code_atms:
    single_atm = data[data["codigo_cajero"] == atm] 
    single_atm = single_atm.sort_values(by=['fecha_transaccion'])
    single_atm_pred = single_atm[single_atm["fecha_transaccion"]<'2024-05-21']

    ts = TimeSeries.from_dataframe(
        single_atm_pred,
        time_col='fecha_transaccion',
        value_cols=['demanda']
    )
    #model = AutoARIMA(start_p=8, max_p=12, start_q=1, n_jobs=-1)
    model = NaiveDrift()
    model.fit(ts)
    pred = model.predict(6)
    pred = pred.pd_dataframe().reset_index()
 
    history = single_atm["demanda"].copy().values[:-6].tolist()
    history.extend(pred['demanda'].values)
    single_atm["demanda"] = history
    
    result = result.append(single_atm)
    print(atm)


result.to_csv("../output/naive_results.csv")


from pycaret.time_series import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = (
    pd.read_csv('./data/Datafest2024_Train.csv')
    .loc[lambda df: (df['codigo_cajero'] == 150)]
    [['demanda']]
)
print(len(data))
# len(data['demanda'].to_list()[-30:])


# data['demanda'] = data['demanda'].interpolate(method='polynomial', order=3).ffill()
# data['demanda'].plot()
plt.show()

# data['demanda'] = np.log(data['demanda'])

df = pd.DataFrame({'demanda': data['demanda'].to_list()[-30:]})
df['demanda'] = np.clip(df['demanda'], 0, 1e12)

df['demanda'] = np.where(df['demanda'] < 0, np.nan, df['demanda'])
# df['demanda'] = df['demanda'].interpolate(method='polynomial', order=3).ffill()


df['demanda'] = df['demanda'].fillna(df['demanda'].mean())

# df['demanda_mov1'] = df['demanda'].shift(1).rolling(window=1).mean()
# df['demanda_mov3'] = df['demanda'].shift(1).rolling(window=3).mean()
# df['demanda_mov7'] = df['demanda'].shift(1).rolling(window=7).mean()


s = setup(df, fh=7, session_id=123, 
          seasonality_type='add', 
          target='demanda', 
          numeric_imputation_exogenous="backfill") # , use_gpu=True

check_stats()

best = compare_models()  # exclude=['snaive']


# # compare models using OOP
# exp.compare_models()

# plot forecast
plot_model(best, plot = 'forecast')

plot_model(best, plot = 'residuals')

save_model(best, 'my_first_model')


loaded_from_disk = load_model('my_first_model')
loaded_from_disk

import pycaret
import darts

final_best = finalize_model(best)

final_best.predict()