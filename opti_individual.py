import pandas as pd
from IPython.display import display
import pulp as lp
import numpy as np

O = 1e9

# Assuming sample as only 1 ATM
df = pd.read_csv('./sample_opti.csv')

atm_type = df['tipo_cajero'].unique()[0]
T = 7
S_0 = df.iloc[0]['saldo_inicial']

if atm_type == 'A':
    P = [np.nan, 1, 1, 0, 0, 1, 0, 0]
    C = 1e6
    R = 0.1 / 100
elif atm_type == 'B':        
    P = [np.nan, 1, 0, 1, 1, 0, 0, 0]
    C = 1e6 + 3e5
    R = 0.15 / 100

df['filling'] = P[1:]
df['filling'] = df['filling'].astype(bool)

df['cumple'] = df['saldo_final'] >= 0.2 * C

W = [-1]
W.extend(df['demanda'].values)
print(W)

x = {t: lp.LpVariable(f'x_{t}', lowBound=0, cat=lp.LpContinuous) for t in range(1, T+1)}
s = {t: lp.LpVariable(f's_{t}', lowBound=0, cat=lp.LpContinuous) for t in range(1, T+1)}

problem = lp.LpProblem('Datathon', lp.LpMinimize)

for t in range(1, T+1):
    # Restriccion 1
    problem += 0.2 * C <= s[t]

    # Restriccion 2 y 3
    if t == 1:
        problem += C >= S_0 + (x[t] - W[t])
        problem += s[t] == S_0 + (x[t] - W[t])
    else:
        problem += C >= s[t-1] + (x[t] - W[t])
        problem += s[t] == s[t-1] + (x[t] - W[t])
        
    # Restriccion 4
    problem += x[t] <= O * P[t]

problem += lp.lpSum([R * P[t] * x[t] for t in range(1, T+1)])
# print(problem)
problem.solve()

print(f'Optimization status: {lp.LpStatus[problem.status]}')

df['abastecimiento_predict'] = [lp.value(x[t]) for t in range(1, T+1)]
df['abastecimiento_predict'] = df['abastecimiento_predict'].astype(int)

saldo_inicial_predict, saldo_final_predict = [], []
demanda = df['demanda'].values
abastecimiento_predict = df['abastecimiento_predict'].values
saldo_acum = S_0
for i in range(T):
    saldo_inicial_predict.append(saldo_acum)
    saldo_acum += abastecimiento_predict[i] - demanda[i]
    saldo_final_predict.append(saldo_acum)

df['saldo_inicial_predict'] = saldo_inicial_predict
df['saldo_final_predict'] = saldo_final_predict
df['cumple_predict'] = df['saldo_final_predict'] >= 0.2 * C

display(df)
display(df[['weekday', 'filling', 'demanda', 'saldo_inicial', 'saldo_final', 'abastecimiento', 'cumple', 'saldo_inicial_predict', 'saldo_final_predict', 'abastecimiento_predict', 'cumple_predict']])


print(f"Costo excel: {df['abastecimiento'].sum() * R}" )
print(f"Costo predict: {df['abastecimiento_predict'].sum() * R}")



df.to_csv('./6b.csv', index=False)
