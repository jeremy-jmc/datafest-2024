import pandas as pd
from IPython.display import display
import pulp as lp
import numpy as np
import copy
from tqdm import tqdm

O = 1e9
T = 7

df = (
    pd.read_csv('./naive_results_v5.csv')
    # .drop(columns=['Unnamed: 0'])
    .assign(
        demanda = lambda df: df['demanda'].astype(int),
        weekday = lambda df: pd.to_datetime(df['fecha_transaccion']).dt.day_of_week
    )
    .sort_values(['codigo_cajero', 'fecha_transaccion'])
    .reset_index(drop=True)
)
display(df)
# df = (
#     pd.read_csv('./nbeats.csv')
#     .drop(columns=['fecha_transaccion'])
# )
# display(df)

# cols = ['demanda_20240521', 'demanda_20240522', 'demanda_20240523', 'demanda_20240524', 'demanda_20240525', 'demanda_20240526', 'demanda_20240527']
# df['demanda_predict'] = df[cols].apply(lambda row: row.values, axis=1)
# df = df.drop(columns=cols)
# df.explode("demanda_predict")


print(df.shape)


def reemplaza_nan(df):
    for i in range(len(df)-1, -1, -1):
        if pd.isna(df.loc[i, 'saldo_final']) and df.loc[i, 'saldo_final'] != df.loc[i, 'saldo_inicial'] and i > 0:
            df.loc[i, 'saldo_inicial'] = df.loc[i-1, 'saldo_final']
    return df


df = (
    reemplaza_nan(df)
    .loc[lambda df: pd.isna(df['abastecimiento']) & pd.isna(df['saldo_final'])]
    .reset_index(drop=True)
)

print(df['codigo_cajero'].value_counts().sort_values())

df_cajeros_semana = df.copy()

display(df_cajeros_semana.head(36))


def generate_opti(df_cajero: pd.DataFrame, verbose=False):
    assert len(df_cajero) == T, "El dataset debe tener solo 7 dias"

    atm_type = df_cajero['tipo_cajero'].unique()[0]
    S_0 = df_cajero.iloc[0]['saldo_inicial']

    if atm_type == 'A':
        P = [np.nan, 1, 1, 0, 0, 1, 0, 0]
        C = 1e6
        R = 0.1 / 100
    elif atm_type == 'B':        
        P = [np.nan, 1, 0, 1, 1, 0, 0, 0]
        C = 1e6 + 3e5
        R = 0.15 / 100
    
    df_cajero['filling'] = P[1:]
    df_cajero['filling'] = df_cajero['filling'].astype(bool)
    df_cajero['cumple'] = df_cajero['saldo_final'] >= 0.2 * C

    W = [-1]
    W.extend(df_cajero['demanda'].values)

    umbral = 0.2
    for min_percentage in list(np.arange(0.2, 0.0, -0.005)) + [0]: # [0.2]:
        x = {t: lp.LpVariable(f'x_{t}', lowBound=0, cat=lp.LpContinuous) for t in range(1, T+1)}
        s = {t: lp.LpVariable(f's_{t}', lowBound=0, cat=lp.LpContinuous) for t in range(1, T+1)}

        print(min_percentage)
        problem = lp.LpProblem('Datathon', lp.LpMinimize)

        threshold = copy.deepcopy(min_percentage)
        umbral = min_percentage
        # Restriccion 1: Los cajeros no caigan por debajo del stock de seguridad (20\% de la capacidad del cajero).
        for t in range(1, T+1):
            problem += s[t] >= lp.value(threshold) * C

        # Restriccion 2: El dinero abastecido al cajero no exceda a su capacidad. Incluyendo la demanda (que puede ser negativa).
        for t in range(1, T+1):
            if t == 1:
                problem += C >= S_0 + (x[t] - W[t])
            else:
                problem += C >= s[t-1] + (x[t] - W[t])

        # Restriccion 3
        for t in range(1, T+1):
            if t == 1:
                problem += s[t] == S_0 + (x[t] - W[t])
            else:
                problem += s[t] == s[t-1] + (x[t] - W[t])

        # Restriccion 4
        for t in range(1, T+1):
            problem += x[t] <= O * P[t]


        problem += lp.lpSum([R * P[t] * x[t] for t in range(1, T+1)])

        # print(problem)
        problem.solve()
        # solver = lp.getSolver('CPLEX_PY')
        # solver = lp.getSolver('PULP_CBC_CMD', timeLimit=10)
        # problem.solve(solver)

        if verbose:
            print(f'Optimization status: {lp.LpStatus[problem.status]}')

        if lp.LpStatus[problem.status] == 'Optimal':
            break
        
    if lp.LpStatus[problem.status] != 'Optimal':
        return -1, pd.DataFrame()

    df_cajero['abastecimiento_predict'] = [lp.value(x[t]) for t in range(1, T+1)]
    df_cajero['abastecimiento_predict'] = df_cajero['abastecimiento_predict'].astype(int)

    saldo_inicial_predict, saldo_final_predict = [], []
    demanda = df_cajero['demanda'].values
    abastecimiento_predict = df_cajero['abastecimiento_predict'].values
    saldo_acum = S_0
    for i in range(T):
        saldo_inicial_predict.append(saldo_acum)
        saldo_acum += abastecimiento_predict[i] - demanda[i]
        saldo_final_predict.append(saldo_acum)

    df_cajero['saldo_inicial_predict'] = saldo_inicial_predict
    df_cajero['saldo_final_predict'] = saldo_final_predict
    df_cajero['cumple_predict'] = df_cajero['saldo_final_predict'] >= (0.2 * C)

    return umbral, df_cajero


atm_ids = df_cajeros_semana['codigo_cajero'].unique()

df_final = pd.DataFrame()

id_fails = []
um = []
for atm in tqdm(atm_ids, total=len(atm_ids), desc='ATM'):
    sub_df = df_cajeros_semana.copy().loc[lambda df: df['codigo_cajero'] == atm]
    umbral, df_opti = generate_opti(sub_df)
    
    if df_opti.empty:
        id_fails.append(atm)
        continue
        # raise RuntimeError(f"Falla en el ATM: {atm}")
    um.append(umbral)
    df_final = pd.concat([df_final, df_opti])

display(df_final.head(36))
print(df_final['cumple_predict'].value_counts())
print(id_fails)
print(len(id_fails))
print(um)


df_final[['fecha_transaccion', 'codigo_cajero', 'tipo_cajero', 'abastecimiento_predict']]

df_output = (
    df_final.pivot(index='codigo_cajero', columns='fecha_transaccion', values='abastecimiento_predict')
    .reset_index().reset_index(drop=True)
)

df_output = (
    df_output.merge(
        df[['codigo_cajero', 'tipo_cajero']].drop_duplicates(keep='first'),
        on='codigo_cajero',
        how='left'
    )
)

df_output["fecha_transaccion"] = "2024-05-20"

date_cols = ["2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-25", "2024-05-26", "2024-05-27"]
main_cols = ["fecha_transaccion", "codigo_cajero", "tipo_cajero"]

df_output = df_output[main_cols + date_cols]
new_date_cols = [f"abastecimiento_{d.replace('-', '')}" for d in date_cols]
df_output.columns = main_cols + new_date_cols

display(df_output)

df_output.to_csv('./Grupo9_Datafest2024_Plantilla_Optimizacion_v5.csv', index=False)
df_output.to_excel('./Grupo9_Datafest2024_Plantilla_Optimizacion_v5.xlsx', index=False)

"""
Grupo9_Datafest2024_Plantilla_Optimizacion.xlsx
Grupo9_Datafest2024_Plantilla_Test.xlsx
Grupo9.pptx
"""