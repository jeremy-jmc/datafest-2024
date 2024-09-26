import pulp as lp
import pandas as pd
from IPython.display import display
import random
import copy
import numpy as np

pd.set_option('display.float_format', lambda x: f'{x:.2f}')

"""
NOTATION:
    m = 1,2, ..., M         # Set of ATM Machines
    t = 1,2, ..., T         # Set of Days for cash flow optimization

    w(m, t)
        Expected cash withdrawal from the m-th ATM during the t-th day
    x(m, t)
        Amount of cash refill for the m-th ATM at the beggining of the t-th day
    s(m, t)
        Amount of cash stock level of the m-th ATM at the end of the t-th day
    d(m, t)
        Standard deviation of cash withdrawal expectation from the m-th ATM during the t-th day
        This can be obtained by comparing the difference between the expected cash withdrawal w(m, t) from the forecasting model and the observed cash withdarawal amount from the historical data
    y(m, t)
        Binary variable that indicates whether the m-th ATM is refilled during the t-th day

    C_s
        Daily interest rate for the idle cash
    C_y
        Refill rate per ATM refill trip
    C_z
        Penalty rate per out-of-cash incident
    
    s_m^0
        Initial cash stock level for the m-th ATM
    B(t)
        Total cash refill budget for the t-th day
    C(m)
        Cash capacity of the m-th ATM
    alpha
        Minimum stock ratio
    N
        Big (positive) number

INNER OPTIMIZATION MODEL

    minimize
        C_s * sum(m, sum(t, s(m, t))) + C_y * sum(m, sum(t, y(m, t)))
    subject to
        s(m, t) = s_m^0 + sum(t, x(m, t) - w(m, t))  for all m, t
        sum(m, x(m, t)) <= B(t)  for all t
        s(m, t) >= alpha * d(m, t)  for all m, t
        s(m, t - 1) + x(m, t) <= C(m)  for all m, t
        x(m, t) <= N * y(m, t)  for all m, t
"""

data = {
    'ATM_1': [98.45, 215.13, 139.75, 141.43, 160.14, 147.97, 95.93, 103.63, 179.92, 114.96],
    'ATM_2': [113.12, 134.02, 36.38, 85.37, 153.36, 158.51, 64.28, 74.31, 122.88, 48.87],
    'ATM_3': [334.81, 504.79, 242.91, 223.05, 266.95, 279.61, 231.92, 319.15, 377.65, 202.92],
    'ATM_4': [126.87, 107.85, 97.30, 109.32, 139.61, 108.10, 137.45, 157.63, 280.11, 119.54],
    'ATM_5': [237.51, 491.85, 154.80, 174.45, 164.85, 239.65, 178.25, 238.95, 365.35, 172.20]
}

data_input = copy.deepcopy(data)
# for k in data_input.keys():
#     data_input[k] = [v * 0.99 for v in data_input[k]]

df = pd.DataFrame(data, index=['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10'])
# df['Total'] = df.sum(axis=1)
# df['Cumulative'] = df['Total'].cumsum() / 1e3
# print(df['Total'].sum() / 10)

C_s = 8.2e-5
C_y = 500.0
C_z = 1000.0
C_m = 1e3
N = 1e3
alpha = 6.1
print(f'{C_s=:.10f} {C_y=:.0f} {C_z=:.0f} {N=:.0f}')

# -----------------------------------------------------------------------------
# Inner loop optimization model
# -----------------------------------------------------------------------------

problem = lp.LpProblem('InnerOptimization', lp.LpMinimize)

# Variables
M = 5
T = 10
B = {t: 2e3 for t in range(1, T+1)}
d = {(m, t): 0 for m in range(1, M+1) for t in range(1, T+1)}   # TODO: Fill with actual values

w = {(m, t): data_input[f'ATM_{m}'][t-1] for m in range(1, M+1) for t in range(1, T+1)}
x = {(m, t): lp.LpVariable(f'x_{m}_{t}', lowBound=0, cat=lp.LpContinuous) for m in range(1, M+1) for t in range(1, T+1)}   # * x(m, t) % 1e3 == 0

hundred_bills = {(m, t): lp.LpVariable(f'hundred_bills_{m}_{t}', lowBound=0, cat=lp.LpInteger) for m in range(1, M+1) for t in range(1, T+1)}
fifty_bills = {(m, t): lp.LpVariable(f'fifty_bills_{m}_{t}', lowBound=0, cat=lp.LpInteger) for m in range(1, M+1) for t in range(1, T+1)}
twenty_bills = {(m, t): lp.LpVariable(f'twenty_bills_{m}_{t}', lowBound=0, cat=lp.LpInteger) for m in range(1, M+1) for t in range(1, T+1)}

for m in range(1, M+1):
    for t in range(1, T+1):
        problem += x[m, t] == (100 * hundred_bills[m, t] + 50 * fifty_bills[m, t] + 20 * twenty_bills[m, t]) / 1000


s = {(m, t): lp.LpVariable(f's_{m}_{t}', lowBound=0, cat=lp.LpContinuous) for m in range(1, M+1) for t in range(1, T+1)}
for _ in range(1, M+1):
    s[_, 0] = 0

y = {(m, t): lp.LpVariable(f'y_{m}_{t}', cat=lp.LpBinary) for m in range(1, M+1) for t in range(1, T+1)}
C = {m: C_m for m in range(1, M+1)}

# Objective function
problem += C_s * lp.lpSum(s.values()) + C_y * lp.lpSum(y.values())

# Constraints
for m in range(1, M+1):
    for t in range(1, T+1):
        # * s(m, t) >= alpha * d(m, t)
        problem += s[m, t] >= alpha * d[m, t]

        # * s(m, t) = s_m^0 + sum(t, x(m, t) - w(m, t))
        problem += s[m, t] == s[m, 0] + lp.lpSum(x[m, v] - w[m, v] for v in range(1, t+1))

        # # * s(m, t-1) + x(m, t) <= C(m)
        # problem += s[m, t-1] + x[m, t] <= C[m]
        
        # * x(m, t) <= N * y(m, t)
        problem += x[m, t] <= C[m] * y[m, t]




for t in range(1, T+1):
    # * sum(m, x(m, t)) <= B(t)
    problem += lp.lpSum(x[v, t] for v in range(1, M+1)) <= B[t]


solver_list = lp.listSolvers(onlyAvailable=True)
print(solver_list)

# lp.pulpTestAll()

solver = lp.getSolver('CPLEX_PY', timeLimit=10)
# solver = lp.getSolver('PULP_CBC_CMD', timeLimit=10)
problem.solve(solver)


results = []
for m in range(1, M+1):
    for t in range(1, T+1):
        results.append({
            'm': m,
            't': t,
            'x': lp.value(x[m, t]), #  / 1e3
            's': lp.value(s[m, t]), #  / 1e3
            'y': lp.value(y[m, t])  #  / 1e3
        })

df_results = pd.DataFrame(results)
display(df_results)

df_results_pivot = (
    df_results.pivot(index='t', columns='m', values='x')
)
# df_results_pivot['daily_cash_flow'] = df_results_pivot.sum(axis=1)
# df_results_pivot['atms_filled'] = df_results_pivot.apply(lambda x: (x > 0).sum(), axis=1)


df_filling = df_results_pivot.copy()
df_filling.index = [f"Day {i+1}" for i in range(10)]
df_withdrawal = df.copy()

display(df_filling)
display(df_withdrawal)

print(f'Optimization status: {lp.LpStatus[problem.status]}')


df_daily_remaining = pd.DataFrame()

for i, col in enumerate(df_withdrawal.columns, start=1):
    df_daily_remaining[col] = df_filling[i].cumsum() - df_withdrawal[col].cumsum()

display(df_daily_remaining)

print(df_results_pivot.sum().sum())
print(df.sum().sum())

# Costo del dinero restante en el día (cash interest cost)
# Sumamos el efectivo que queda cada día y multiplicamos por la tasa de interés
cash_interest_cost = C_s * df_daily_remaining[df_daily_remaining > 0].sum().sum()

# Costo de recargas (basado en cuántas veces se recargó cada cajero)
# Suponiendo que cualquier valor mayor a 0 en df_results_pivot implica una recarga
cash_refill_cost = C_y * np.count_nonzero(df_filling)

# Costo de fuera de caja (out-of-cash penalty cost)
# Sumamos los valores negativos (cuando el cajero se queda sin efectivo) y aplicamos la penalización
out_of_cash_penalty_cost = C_z * (-df_daily_remaining[df_daily_remaining < 0].sum().sum())


total_optimization_cost = cash_interest_cost + cash_refill_cost + out_of_cash_penalty_cost

print(f"{cash_interest_cost=} {cash_refill_cost=} {out_of_cash_penalty_cost=}")
print(f"{total_optimization_cost=}")



# -----------------------------------------------------------------------------
# Outer optimization loop model
# -----------------------------------------------------------------------------

# Step 1: Initialize alpha and other variables
alpha_init = copy.deepcopy(alpha)  # Initial stock ratio
max_iterations = 1000
tol = 1e-3  # Convergence tolerance
learning_rate = 0.01  # Learning rate for updating alpha
alpha = alpha_init

# Placeholder for the function to compute the overall cost from Monte Carlo Simulation
def monte_carlo_simulation(x, w_tilde, s, C_s, C_y, C_z):
    # Simulate costs based on stochastic withdrawals w_tilde
    # Returns the overall ATM cash management cost
    
    # Use lp.value() to extract the numerical values from the LpVariables
    total_cost = C_s * np.sum([lp.value(v) for v in s.values()]) + C_y * np.sum([lp.value(y) for y in x.values() if lp.value(y) > 0])
    
    out_of_cash_penalty = 0
    
    for m, t in w_tilde.keys():
        s_tilde = lp.value(s[(m, t)]) - w_tilde[(m, t)]  # Simulate stock level after stochastic withdrawals
        if s_tilde < 0:
            out_of_cash_penalty += C_z  # Apply penalty for out-of-cash incidents
    
    # print(f"{out_of_cash_penalty=}")
    return total_cost + out_of_cash_penalty

# Step 2: Outer Optimization Loop
for iteration in range(max_iterations):
    print(f"Iteration {iteration}: Current alpha = {alpha}")
    
    # Step 2.1: Run inner loop optimization with the current alpha
    # (Assuming the inner optimization loop function returns x, s)
    # x, s = inner_optimization(alpha)
    
    # Step 2.2: Perform Monte Carlo simulations
    w_tilde = { (m, t): np.random.normal(w[(m, t)], d[(m, t)]) for m, t in w.keys() }
    total_cost = monte_carlo_simulation(x, w_tilde, s, C_s, C_y, C_z)
    
    print(f"Total cost: {total_cost}")
    
    # Step 2.3: Update alpha using a line search strategy (here using a simple gradient descent update)
    # Calculate the gradient (assuming some derivative computation or estimation)
    gradient = -1.0  # Placeholder for the actual gradient calculation
    
    # Update alpha
    alpha_new = alpha - learning_rate * gradient
    
    # Step 2.4: Check for convergence
    if abs(alpha_new - alpha) < tol:
        print(f"Converged at alpha = {alpha_new}")
        break
    
    alpha = alpha_new


"""
TODO: jugar con un porcentaje de no atencion de cajeros de 0 a 1% y ver como se comporta el costo total
"""