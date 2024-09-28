import pandas as pd
from IPython.display import display
import random

df_train = pd.read_csv('./data/Datafest2024_Train.csv')

print(df_train.isna().sum())

display(
    df_train.groupby(['codigo_cajero'])
    .agg({'fecha_transaccion': 'nunique', 'tipo_cajero': 'unique'})
)

df_pivot = (
    df_train[['codigo_cajero', 'fecha_transaccion']]
    .assign(value=1)
    .pivot(index='codigo_cajero', columns='fecha_transaccion', values='value')
)
print(df_pivot.isna().sum().sort_values())


print(df_train['codigo_cajero'].drop_duplicates(keep='first').sort_values())


print(
    df_train[['codigo_cajero', 'tipo_cajero']].drop_duplicates(keep='first')
    ['tipo_cajero'].value_counts()
)


df_train.pivot(index='fecha_transaccion', columns='codigo_cajero', values=['saldo_inicial', 'demanda', 'abastecimiento'])

df_train['fecha'] = pd.to_datetime(df_train['fecha_transaccion'], format='%Y%m%d')

df_train['year'] = df_train['fecha'].dt.year
df_train['week'] = df_train['fecha'].dt.isocalendar().week
df_train['weekday'] = df_train['fecha'].dt.day_of_week.replace({
    0: 'L',
    1: 'M', 
    2: 'X',
    3: 'J',
    4: 'V',
    5: 'S',
    6: 'D'
})

display(df_train.head(50))


df_train = df_train.loc[lambda df: ((df['year'] >= 2023) & (df['week'] >= 23)) | (df['year'] == 2024)]

# df_train.to_csv('./cajeros_semana.csv', index=False)

display(
    df_train[['year', 'week']]
    .assign(freq=1)
    .groupby(['year', 'week'], as_index=False)
    .agg({'freq': 'sum'})
    .assign(freq = lambda df: df['freq']/700)
    # .loc[lambda df: df['freq'] == 7]
)



# -----------------------------------------------------------------------------
# Get random ATM data
# -----------------------------------------------------------------------------

atm_codes = df_train['codigo_cajero'].unique()
random_atm = random.choice(atm_codes)

random_atm = 6
df_atm = (
    df_train.loc[lambda df: df['codigo_cajero'] == random_atm]
    .reset_index(drop=True)
    .drop(columns=['fecha_transaccion'])
)
df_atm

display(df_atm[['year', 'week']].assign(freq=1).groupby(['year', 'week']).agg({'freq': 'sum'}))


random_year = random.choice([2023, 2024])
random_week = random.choice(df_atm.loc[lambda df: df['year'] == random_year]['week'].unique())

random_year = 2023
random_week = 23

idx_init = max(0, df_atm.loc[lambda df: (df['year'] == random_year) & (df['week'] == random_week)].index[0] - 1)
idx_fin = idx_init + 7

display(df_atm.iloc[idx_init:idx_fin+1])

display(df_atm.loc[lambda df: (df['year'] == random_year) & (df['week'] == random_week)])

df_atm.loc[lambda df: (df['year'] == random_year) & (df['week'] == random_week)].to_csv('./sample_opti.csv', index=False)

