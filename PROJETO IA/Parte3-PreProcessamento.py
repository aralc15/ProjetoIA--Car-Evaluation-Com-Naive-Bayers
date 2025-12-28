import pandas as pd
import os
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ==========================================
# 1. Config e definicao
# ==========================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
DADOS = "data"
FILE_PATH = os.path.join(DADOS, "car.data")

COLUMNS = [
    "buying",      # preço de compra
    "maint",       # custo de manutenção
    "doors",       # número de portas
    "persons",     # capacidade de pessoas
    "lug_boot",    # tamanho do porta-malas
    "safety",      # nível de segurança
    "class"        # classe alvo
]

# ==========================================
# 2. DOWNLOAD DO DATASET (AM- Naive Bayes)
# ==========================================
os.makedirs(DADOS, exist_ok=True) 

if not os.path.exists(FILE_PATH):
    print("Baixando o conjunto de dados Car Evaluation...")
    urllib.request.urlretrieve(DATA_URL, FILE_PATH)
    print("Download concluído.")
else:
    print("ARQUIVO JÁ EXISTENTE. CARREGANDO....")

# ==========================================
# 3. VERIFICAÇÃO DE INTEGRIDADE DOS DADOS
# ==========================================
df = pd.read_csv(FILE_PATH, header=None, names=COLUMNS)

print("\n" + "="*60)
print("VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
print("="*60)
print(f"Dimensão do DataFrame: {df.shape}")
print(f"Duplicatas encontradas: {df.duplicated().sum()}")
print(f"Valores nulos totais: {df.isnull().sum().sum()}")
print("-"*60)

# ==========================================
# 4. ANÁLISE EXPLORATÓRIA VISUAL (EDA) 
# ==========================================
print("\n" + "="*60)
print("ANÁLISE EXPLORATÓRIA VISUAL (EDA)")
print("="*60)

# Configuração de estilo
sns.set_style("whitegrid")

# 4.1 Visualizar Balanceamento da Variável Alvo
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='class', data=df, order=df['class'].value_counts().index, palette='viridis')
plt.title('Distribuição da Variável Alvo (class)')
plt.xlabel('Classes')
plt.ylabel('Contagem')

# Adicionar porcentagens nas barras
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()

print("Observação: Note o desbalanceamento severo na classe 'unacc' (inaceitável).")

# ==========================================
# 5. Mapeamento e codificao ordinal
# ==========================================
print("\n" + "="*60)
print("MAPEAMENTO DAS VARIÁVEIS ORDINAIS")
print("="*60)
print("Transformando categorias de texto em números respeitando a ordem de grandeza.")

# Definiçao dos Mapeamentos (Sugestão Formal)
map_buying_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
map_doors = {'2': 2, '3': 3, '4': 4, '5more': 5}
map_persons = {'2': 2, '4': 4, 'more': 5}
map_lug = {'small': 0, 'med': 1, 'big': 2}
map_safety = {'low': 0, 'med': 1, 'high': 2}

# Aplicando as transformações no DataFrame
# Criamos uma cópia para preservar o original se necessário, ou aplicamos direto
df_encoded = df.copy()

df_encoded['buying'] = df_encoded['buying'].map(map_buying_maint)
df_encoded['maint'] = df_encoded['maint'].map(map_buying_maint)
df_encoded['doors'] = df_encoded['doors'].map(map_doors)
df_encoded['persons'] = df_encoded['persons'].map(map_persons)
df_encoded['lug_boot'] = df_encoded['lug_boot'].map(map_lug)
df_encoded['safety'] = df_encoded['safety'].map(map_safety)

# A variável alvo também pode ser mapeada se desejado, mas geralmente deixa-se para o modelo
# map_class = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
# df_encoded['class'] = df_encoded['class'].map(map_class)

print("\nVisualização pós-codificação (5 primeiras linhas):")
print("-"*60)
print(df_encoded.head())
print("-"*60)
print("\nNovos tipos de dados (agora numéricos):")
print(df_encoded.dtypes)


# 1. Recuperando o DataFrame (Simulação do que as Pessoas 1 e 2 entregaram)
# (Aqui usei a estrutura de colunas e mapeamentos definidos no código deles)
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# 2. Definição dos Mapeamentos Ordinais (Baseado no Passo 2)
map_buying_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3} 
map_doors = {'2': 2, '3': 3, '4': 4, '5more': 5}
map_persons = {'2': 2, '4': 4, 'more': 5}
map_lug = {'small': 0, 'med': 1, 'big': 2}
map_safety = {'low': 0, 'med': 1, 'high': 2}
map_class = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

# 3. Transformação dos Dados (Codificação Ordinal)
df_encoded = df.copy()
df_encoded['buying'] = df_encoded['buying'].map(map_buying_maint)
df_encoded['maint'] = df_encoded['maint'].map(map_buying_maint)
df_encoded['doors'] = df_encoded['doors'].map(map_doors)
df_encoded['persons'] = df_encoded['persons'].map(map_persons)
df_encoded['lug_boot'] = df_encoded['lug_boot'].map(map_lug)
df_encoded['safety'] = df_encoded['safety'].map(map_safety)
df_encoded['class'] = df_encoded['class'].map(map_class)

# 4. Divisão em Atributos (X) e Alvo (y)
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# 5. Divisão em Treino e Teste (O "Gran Finale" do Passo 3)
# Usamos 30% para teste e 70% para treino. 
# random_state=42 garante que a divisão seja sempre a mesma para todos os colegas.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Passo 3 concluído com sucesso!")
print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")
