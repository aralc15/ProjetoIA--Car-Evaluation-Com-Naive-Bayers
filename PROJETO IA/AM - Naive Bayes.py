
import pandas as pd
import os
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data" # Define o link oficial da UCI
DADOS = "data" #PASTA CRIDA
FILE_PATH = os.path.join(DADOS, "car.data")

# Nomes das colunas conforme documentação da UCI
COLUMNS = [
    "buying",      # preço de compra
    "maint",       # custo de manutenção
    "doors",       # número de portas
    "persons",     # capacidade de pessoas
    "lug_boot",    # tamanho do porta-malas
    "safety",      # nível de segurança
    "class"        # classe alvo
]

# 2. Download do dataset

os.makedirs(DADOS, exist_ok=True) 

if not os.path.exists(FILE_PATH):
    print("Baixando o conjunto de dados Car Evaluation...")
    urllib.request.urlretrieve(DATA_URL, FILE_PATH)
    print("Download concluído.")
else:
    print("CARREGANDO....")

# 3. Carregamento do dataset
df = pd.read_csv(FILE_PATH, header=None, names=COLUMNS)

# 4. Verificação de integridade
print("\n" + "="*60)
print("VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
print("="*60)
print(f"\nDimensão do DataFrame (linhas, colunas): {df.shape}")
print("\nVisualização inicial (5 primeiras linhas):")
print("-"*60)
print(df.head())
print("-"*60)
print("\nTipos de dados por coluna:")
print("-"*60)
print(df.dtypes)
print("-"*60)

# 5. Analise básica dos dados
print("\n" + "="*60)
print("ANÁLISES BÁSICAS")
print("="*60)

print("\nValores ausentes por coluna:")
print("-"*60)
print(df.isnull().sum())
print("-"*60)

print("\nRegistros duplicados:")
print("-"*60)
print(df.duplicated().sum())
print("-"*60)

print("\nDistribuição da variável alvo (class):")
print("-"*60)
print(df["class"].value_counts())
print("-"*60)