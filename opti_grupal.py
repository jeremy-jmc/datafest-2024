import pandas as pd
from IPython.display import display
import pulp as lp
import numpy as np
import copy
from tqdm import tqdm
import logging
# logging.basicConfig(level=logging.ERROR)

O = 1e9
T = 7


df = pd.read_csv('./cajeros_semana.csv')
print(df.shape)

df_cajeros_semana = df.copy()

df_cajeros_a_more1M = df_cajeros_semana.loc[lambda df: (df['tipo_cajero'] == 'A') & (df['saldo_final'] >= 1e6)]
# e.g. 365


tripletas = df_cajeros_a_more1M[['tipo_cajero', 'year', 'week']].drop_duplicates(keep='first').values.tolist()

df_cajeros_semana['delete'] = (
    df_cajeros_semana[['tipo_cajero', 'year', 'week']].apply(lambda tup: list(tup) in tripletas, axis=1)
)

df_cajeros_semana = (
    df_cajeros_semana
    .loc[lambda df: df['delete'] == False]
)


def generate_opti(df_cajero: pd.DataFrame, verbose=False):
    assert len(df_cajero) == 7, "El dataset debe tener solo 7 dias"

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
    for min_percentage in [0.2]: # list(np.arange(0.2, 0.0, -0.01)) + [0]:
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
        # solver = lp.getSolver('CPLEX_PY', timeLimit=10)
        # # solver = lp.getSolver('PULP_CBC_CMD', timeLimit=10)
        # problem.solve(solver)
        problem.solve()

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
    df_cajero['cumple_predict'] = df_cajero['saldo_final_predict'] >= 0.2 * C

    return umbral, df_cajero


# display(df_cajeros_semana[['codigo_cajero', 'year', 'week']].drop_duplicates(keep='first'))

atm_ids = df_cajeros_semana['codigo_cajero'].unique()

df_final = pd.DataFrame()

id_fails = []
for atm in tqdm(atm_ids, total=len(atm_ids), desc='ATM'):
    sub_df = df_cajeros_semana.copy().loc[lambda df: df['codigo_cajero'] == atm]
    year_weeks = sub_df[['year', 'week']].values

    for ix, (y, w) in enumerate(year_weeks): # tqdm(year_weeks, total=len(year_weeks), desc='Year_Week'):
        if ix <= 5:
            continue
        umbral, df_opti = generate_opti(sub_df.loc[lambda df: (df['year'] == y) & (df['week'] == w)].copy())
        if df_opti.empty:
            id_fails.append(atm)
        df_final = pd.concat([
            df_final, df_opti
        ])
    # display(df_opti)

display(df_final)
print(id_fails)
print(len(id_fails))

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

atm = 365
sub_df = df_cajeros_semana.copy().loc[lambda df: df['codigo_cajero'] == atm]
year_weeks = sub_df[['year', 'week']].values

for y, w in year_weeks: # tqdm(year_weeks, total=len(year_weeks), desc='Year_Week'):
    df_check = sub_df.loc[lambda df: (df['year'] == y) & (df['week'] == w)].copy()
    umbral, df_opti = generate_opti(df_check, True)
    break


display(df_check)

print(f"Using {umbral} in ATM {atm}")
display(df_opti)
