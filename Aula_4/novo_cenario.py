import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Carrega os dados gerados
df = pd.read_csv("telemetria_servidores.csv")  # ou telemetria.csv

# Define as colunas corretas para X e y
X = df[["msgs_per_sec", "latencia_ms", "uso_cpu_pct"]]
y = df["custo_real"]

# Cria pipeline com normalização e regressão linear
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Treina o modelo
pipeline.fit(X, y)

# Novo cenário que queremos testar
novo_cenario = pd.DataFrame({
    "msgs_per_sec": [1350],
    "latencia_ms": [280],
    "uso_cpu_pct": [70]
})

# Faz a previsão
custo_previsto = pipeline.predict(novo_cenario)[0]
print(f"Custo Previsto para o novo cenário: R$ {custo_previsto:.2f}")