import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# DATASET
data = {
    'tempo_conversa_min': [5, 12, 2, 20, 15, 8, 30, 10, 5, 25],
    'mensagens_trocadas': [10, 25, 4, 45, 30, 12, 60, 18, 9, 50],
    'custo_ads_real': [15.5, 32.0, 8.0, 55.0, 40.5, 20.0, 80.0, 25.0, 14.0, 65.0]
}

df = pd.DataFrame(data)

# --- DESAFIO DE REGRESSÃO MÚLTIPLA ---

# 1️⃣ Definindo X (duas variáveis independentes)
X = df[['tempo_conversa_min', 'mensagens_trocadas']]

# 2️⃣ Definindo y (variável alvo)
y = df['custo_ads_real']

# 3️⃣ Criando o modelo
model = LinearRegression()

# 4️⃣ Treinando o modelo
model.fit(X, y)

# 5️⃣ Fazendo a previsão
previsao = model.predict([[15, 35]])

print("Previsão de custo para 15min e 35 mensagens:", previsao[0])