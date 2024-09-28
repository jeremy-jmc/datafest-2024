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