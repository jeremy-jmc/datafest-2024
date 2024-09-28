import pandas as pd
from IPython.display import display

df = (
    pd.read_excel('./data/Datafest2024_Plantilla_Optimizacion.xlsx')
    .dropna(subset=['codigo_cajero', 'fecha_transaccion'], how='any')
    .assign(
        fecha_transaccion = lambda x: pd.to_datetime(x['fecha_transaccion'].astype(int), format='%Y%m%d') - pd.Timedelta(days=1),
        codigo_cajero = lambda x: x['codigo_cajero'].astype(int)
    )
    .loc[lambda df: df['fecha_transaccion'] == '2024-05-20']
    .sort_values(by=['codigo_cajero'])
    .reset_index(drop=True)
)
display(df)
