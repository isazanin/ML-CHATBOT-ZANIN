import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

df = pd.read_csv('mensagens_suporte.csv')

# NORMALIZAÇÃO
df['texto_limpo'] = df['texto'].str.lower()

# FEATURES
df['qtd_exclamacao'] = df['texto'].str.count('!')

palavras_alerta = ['atraso', 'cancelar', 'chegou', 'cadê', 'agora']
df['tem_palavra_alerta'] = df['texto_limpo'].apply(
    lambda x: 1 if any(p in x for p in palavras_alerta) else 0
)

df['tamanho_msg'] = df['texto'].str.len()

# 1. Preparação (Usando as features da Aula 02)
# Supondo que o df já tenha as colunas: 'qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg'
X = df[['qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg']]
y = df['rotulo'] # 1: Alta Prioridade, 0: Comum

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Treinamento
clf = DecisionTreeClassifier(max_depth=3) # Limitamos a profundidade para ser legível
clf.fit(X_train, y_train)

# 3. Visualização da Lógica (Explicação para os alunos)
# Diferente de uma "caixa preta", a Árvore nos mostra as regras que criou
regras = export_text(clf, feature_names=['Exclamações', 'Palavra Alerta', 'Tamanho'])
print("--- Regras Decididas pelo Modelo ---")
print(regras)
