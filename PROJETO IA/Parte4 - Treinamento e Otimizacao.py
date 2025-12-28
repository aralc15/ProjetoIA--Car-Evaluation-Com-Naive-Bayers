import pandas as pd
import os
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==========================================
# 1. CONFIGURAÇÃO E DEFINIÇÃO
# ==========================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
DADOS = "data"
FILE_PATH = os.path.join(DADOS, "car.data")

COLUMNS = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]

# ==========================================
# 2. DOWNLOAD DO DATASET
# ==========================================
os.makedirs(DADOS, exist_ok=True)

if not os.path.exists(FILE_PATH):
    print("Baixando o conjunto de dados Car Evaluation...")
    urllib.request.urlretrieve(DATA_URL, FILE_PATH)
    print("Download concluído.")
else:
    print("ARQUIVO JÁ EXISTENTE. CARREGANDO...")

# ==========================================
# 3. VERIFICAÇÃO DE INTEGRIDADE
# ==========================================
df = pd.read_csv(FILE_PATH, header=None, names=COLUMNS)

print("\n" + "="*60)
print("VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
print("="*60)
print(f"Dimensão do DataFrame: {df.shape}")
print(f"Duplicatas encontradas: {df.duplicated().sum()}")
print(f"Valores nulos totais: {df.isnull().sum().sum()}")

# ==========================================
# 4. ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================
sns.set_style("whitegrid")

plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x='class',
    data=df,
    order=df['class'].value_counts().index,
    palette='viridis'
)
plt.title('Distribuição da Variável Alvo')
plt.xlabel('Classe')
plt.ylabel('Contagem')

total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(
        percentage,
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.show()

print("Observação: Forte desbalanceamento da classe 'unacc'.")

# ==========================================
# 5. MAPEAMENTO E CODIFICAÇÃO ORDINAL
# ==========================================
map_buying_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
map_doors = {'2': 2, '3': 3, '4': 4, '5more': 5}
map_persons = {'2': 2, '4': 4, 'more': 5}
map_lug = {'small': 0, 'med': 1, 'big': 2}
map_safety = {'low': 0, 'med': 1, 'high': 2}

df_encoded = df.copy()

df_encoded['buying'] = df_encoded['buying'].map(map_buying_maint)
df_encoded['maint'] = df_encoded['maint'].map(map_buying_maint)
df_encoded['doors'] = df_encoded['doors'].map(map_doors)
df_encoded['persons'] = df_encoded['persons'].map(map_persons)
df_encoded['lug_boot'] = df_encoded['lug_boot'].map(map_lug)
df_encoded['safety'] = df_encoded['safety'].map(map_safety)

# ==========================================
# 6. DEFINIÇÃO DO PROBLEMA (BINÁRIO)
# ==========================================
# 1 = oportunidade (good, vgood)
# 0 = não oportunidade (unacc, acc)

df_focus = df_encoded.copy()
df_focus['class'] = df_focus['class'].apply(
    lambda x: 1 if x in ['good', 'vgood'] else 0
)

X = df_focus.drop('class', axis=1)
y = df_focus['class']

# ==========================================
# 7. DIVISÃO TREINO / TESTE (70 / 30)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nDivisão realizada com sucesso!")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# ==========================================
# 8. IMPUTAÇÃO (ROBUSTEZ DO PIPELINE)
# ==========================================
imputer = SimpleImputer(strategy="most_frequent")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ==========================================
# 9. TREINAMENTO DO MODELO (AJUSTADO)
# ==========================================
# 🔴 MUDANÇA IMPORTANTE:
# Forçando o modelo a considerar a classe oportunidade

model = CategoricalNB(
    alpha=1.0,
    class_prior=[0.80, 0.20]
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.60
y_pred = (y_proba >= threshold).astype(int)


# ==========================================
# 11. AVALIAÇÃO DO MODELO
# ==========================================
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_pos = f1_score(y_test, y_pred, pos_label=1)

print("\n" + "="*60)
print("AVALIAÇÃO DO MODELO (AJUSTADO AO OBJETIVO)")
print("="*60)
print(f"Acurácia: {acc:.4f}")
print(f"Acurácia Balanceada: {bal_acc:.4f}")
print(f"F1-score Macro: {f1_macro:.4f}")
print(f"F1-score (Oportunidades): {f1_pos:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# ==========================================
# 12. MATRIZ DE CONFUSÃO
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Não oportunidade', 'Oportunidade'],
    yticklabels=['Não oportunidade', 'Oportunidade']
)
plt.title('Matriz de Confusão - Naive Bayes Ajustado')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.show()
