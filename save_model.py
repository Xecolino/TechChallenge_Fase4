import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # Importação adicionada para formatar o eixo Y do gráfico de importância

# --- 1. Carregamento e Pré-processamento dos Dados ---
df = pd.read_csv('Dados Históricos - Ibovespa.csv', sep=',')
df.columns = df.columns.str.strip()
df = df.rename(columns={'Data': 'data', 'Último': 'ultimo','Abertura': 'abertura', 'Máxima': 'maxima', 'Mínima': 'minima', 'Vol.': 'vol_', 'Var%': 'var_'})

def convert_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%d.%m.%Y')
    except ValueError:
        excel_epoch = datetime(1899, 12, 30)
        return excel_epoch + timedelta(days=float(date_str))

df['data'] = df['data'].apply(convert_date)
df = df.sort_values(by='data').reset_index(drop=True)

def clean_numeric_column(series):
    series = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    if series.name == 'vol_':
        series = series.str.replace('M', 'e6', regex=False).str.replace('B', 'e9', regex=False)
        return pd.to_numeric(series, errors='coerce')
    else:
        return pd.to_numeric(series, errors='coerce')

df['var_'] = df['var_'].str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
df['var_'] = pd.to_numeric(df['var_'], errors='coerce')

numeric_cols = ["ultimo", "abertura", "maxima", "minima", "vol_"]
for col in numeric_cols:
    df[col] = clean_numeric_column(df[col])

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 2. Engenharia de Atributos ---
df['MA5'] = df['ultimo'].rolling(window=5).mean()
df['MA10'] = df['ultimo'].rolling(window=10).mean()
df['MA20'] = df['ultimo'].rolling(window=20).mean()

df['ultimo_lag1'] = df['ultimo'].shift(1)
df['abertura_lag1'] = df['abertura'].shift(1)
df['maxima_lag1'] = df['maxima'].shift(1)
df['minima_lag1'] = df['minima'].shift(1)
df['vol_lag1'] = df['vol_'].shift(1)
df['var_lag1'] = df['var_'].shift(1)

df['range_dia_anterior'] = df['maxima'].shift(1) - df['minima'].shift(1)
df['abertura_fechamento_diff_lag1'] = df['abertura'].shift(1) - df['ultimo'].shift(1)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 3. Criação da Variável Target ---
df['target'] = (df['ultimo'].shift(-1) > df['ultimo']).astype(int)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 4. Divisão dos Dados ---
test_size_days = 30 # Ajuste este valor se necessário, com base na sua depuração anterior
X = df.drop(['data', 'target',"ultimo","MA10","MA20","ultimo_lag1","minima_lag1","vol_lag1","var_lag1","maxima_lag1","range_dia_anterior","MA5"], axis=1)
y = df['target']

# Verificação para garantir que X_train não está vazio
if df.shape[0] <= test_size_days:
    print(f"ERRO CRÍTICO: O número de amostras disponíveis ({df.shape[0]}) é menor ou igual a 'test_size_days' ({test_size_days}).")
    print("Não há dados suficientes para criar um conjunto de treinamento não vazio. Por favor, reduza o valor de 'test_size_days' ou forneça mais dados históricos.")
    exit()

X_train = X.iloc[:-test_size_days]
X_test = X.iloc[-test_size_days:]
y_train = y.iloc[:-test_size_days]
y_test = y.iloc[-test_size_days:]

# --- 5. Treinamento do Random Forest ---
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

# --- 6. SALVAR O MODELO CORRETAMENTE ---
joblib.dump(rf_model, 'forest.joblib')
print("Modelo Random Forest salvo com sucesso em 'forest.joblib'")

# --- 7. Gerar e Salvar a Matriz de Confusão ---
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Baixa (0)', 'Alta (1)'])
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
ax_cm.set_title('Matriz de Confusão do Modelo (Conjunto de Teste)')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close(fig_cm)
print("Matriz de Confusão salva com sucesso em 'confusion_matrix.png'")

# --- 8. Gerar e Salvar o Gráfico de Importância das Features ---
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
feature_importances.plot(kind='barh', ax=ax_fi, color='skyblue')
ax_fi.set_title('Importância das Features no Modelo Random Forest')
ax_fi.set_xlabel('Importância')
ax_fi.set_ylabel('Feature')
ax_fi.invert_yaxis() # Para ter a feature mais importante no topo
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close(fig_fi)
print("Gráfico de Importância das Features salvo com sucesso em 'feature_importance.png'")


# --- Opcional: Verificar se o arquivo foi salvo corretamente ---
loaded_model_test = joblib.load('forest.joblib')
print(f"Tipo do modelo carregado para teste: {type(loaded_model_test)}")
