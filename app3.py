import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
from datetime import datetime, timedelta

# --- 1. Configurações Iniciais da Aplicação Streamlit ---
st.set_page_config(
    page_title="Previsão IBOVESPA - Tech Challenge",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Previsão de Tendência do IBOVESPA")
st.markdown("""
    Esta aplicação interativa prevê se o IBOVESPA fechará em alta (1) ou baixa (0) no dia seguinte,
    utilizando um modelo de Machine Learning Random Forest treinado.
""")

# --- 2. Carregamento do Modelo e Dados Históricos ---
@st.cache_resource
def load_model(model_path):
    """Carrega o modelo treinado."""
    if not os.path.exists(model_path):
        st.error(f"Erro: O arquivo do modelo '{model_path}' não foi encontrado. "
                 "Certifique-se de que o arquivo 'forest.joblib' está no mesmo diretório do 'app.py'.")
        st.stop()
        return None

    try:
        raw_loaded_object = joblib.load(model_path)
        
        if isinstance(raw_loaded_object, str):
            st.error(f"Erro: O arquivo '{model_path}' foi carregado como uma **string** em vez de um modelo. "
                     "Isso geralmente indica que o arquivo está corrompido ou não foi salvo corretamente "
                     "como um modelo joblib. Por favor, verifique se você salvou seu modelo usando "
                     "`joblib.dump(seu_modelo, 'forest.joblib')` e se o arquivo não foi alterado.")
            st.stop()
            return None
        
        model = raw_loaded_object
        st.success(f"Modelo '{model_path}' carregado com sucesso. Tipo esperado: {type(model)}")
        return model
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo '{model_path}': {type(e).__name__}: {e}. "
                 "Verifique a integridade do arquivo 'forest.joblib'.")
        st.stop()
        return None

@st.cache_data
def load_historical_data(file_path):
    """Carrega e pré-processa os dados históricos do IBOVESPA."""
    try:
        df = pd.read_csv(file_path, sep=',')
        
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Data': 'data',
            'Último': 'ultimo',
            'Abertura': 'abertura',
            'Máxima': 'maxima',
            'Mínima': 'minima',
            'Vol.': 'vol_',
            'Var%': 'var_'
        }
        df = df.rename(columns=rename_map)
        
        if 'data' not in df.columns:
            raise KeyError("A coluna 'data' não foi encontrada após o pré-processamento. Verifique o cabeçalho do seu CSV para 'Data' ou 'data'.")

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

        if 'var_' in df.columns:
            df['var_'] = df['var_'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
            df['var_'] = pd.to_numeric(df['var_'], errors='coerce')

        numeric_cols_to_clean = [col for col in ["ultimo", "abertura", "maxima", "minima", "vol_"] if col in df.columns]
        for col in numeric_cols_to_clean:
            df[col] = clean_numeric_column(df[col])

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['ultimo_lag1_temp'] = df['ultimo'].shift(1)
        df['abertura_lag1_temp'] = df['abertura'].shift(1)
        df['maxima_lag1_temp'] = df['maxima'].shift(1)
        df['minima_lag1_temp'] = df['minima'].shift(1)
        df['vol_lag1_temp'] = df['vol_'].shift(1)
        df['var_lag1_temp'] = df['var_'].shift(1)
        df['range_dia_anterior_temp'] = df['maxima'].shift(1) - df['minima'].shift(1)
        df['abertura_fechamento_diff_lag1_temp'] = df['abertura'].shift(1) - df['ultimo'].shift(1)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de dados históricos '{file_path}' não foi encontrado. Certifique-se de que está no mesmo diretório.")
        st.stop()
        return None
    except Exception as e:
        st.error(f"Erro ao carregar ou pré-processar os dados históricos: {type(e).__name__}: {e}")
        st.stop()
        return None

# --- Opcional: Função para Salvar Log ---
def save_log(data, filename='prediction_log.csv'):
    """Salva os dados de log em um arquivo CSV."""
    log_df = pd.DataFrame([data])
    if not os.path.exists(filename):
        log_df.to_csv(filename, index=False)
    else:
        log_df.to_csv(filename, mode='a', header=False, index=False)


# Caminhos dos arquivos
MODEL_PATH = 'forest.joblib'
DATA_PATH = 'Dados Históricos - Ibovespa.csv'
CM_IMAGE_PATH = 'confusion_matrix.png'
FI_IMAGE_PATH = 'feature_importance.png' # Novo caminho para a imagem de importância das features

model = load_model(MODEL_PATH)
historical_df = load_historical_data(DATA_PATH)

if model is None or historical_df is None:
    st.error("A aplicação não pode iniciar devido a erros no carregamento do modelo ou dados históricos. Por favor, verifique os logs acima.")
    st.stop()

# --- 3. Painel de Métricas de Validação do Modelo ---
st.header("📊 Performance do Modelo")
st.info(f"**Modelo Utilizado:** Random Forest Classifier")
st.metric(label="Acurácia de Validação", value="80.00%")
st.markdown("""
    > *Lembre-se: Esta acurácia foi obtida durante a fase de treinamento e validação em um conjunto de teste 
    apresentado no Tech Challenge 02.*
""")

# --- Adicionar Matriz de Confusão ---
st.subheader("Matriz de Confusão")
st.markdown("""
    A matriz de confusão é uma ferramenta essencial para avaliar a performance de um modelo de classificação.
    Ela mostra o número de previsões corretas e incorretas, categorizadas por classe.
""")

if os.path.exists(CM_IMAGE_PATH):
    st.image(CM_IMAGE_PATH, caption='Matriz de Confusão do Modelo (Conjunto de Teste)', use_column_width=True)
    st.markdown("""
        - **Verdadeiro Negativo (TN):** Previsões corretas de "Baixa (0)". O modelo previu baixa e o IBOVESPA realmente caiu.
        - **Falso Positivo (FP):** Previsões incorretas de "Alta (1)". O modelo previu alta, mas o IBOVESPA caiu (Erro Tipo I).
        - **Falso Negativo (FN):** Previsões incorretas de "Baixa (0)". O modelo previu baixa, mas o IBOVESPA subiu (Erro Tipo II).
        - **Verdadeiro Positivo (TP):** Previsões corretas de "Alta (1)". O modelo previu alta e o IBOVESPA realmente subiu.
    """)
else:
    st.warning("Arquivo 'confusion_matrix.png' não encontrado. Por favor, execute o script de treinamento (`save_model.py`) para gerá-lo.")

# --- Adicionar Gráfico de Importância das Features ---
st.subheader("Importância das Features")
st.markdown("""
    Este gráfico mostra quais atributos (features) foram mais relevantes para o modelo Random Forest
    ao tomar suas decisões de previsão. Features com maior importância contribuíram mais para a redução
    da impureza nas árvores de decisão.
""")
if os.path.exists(FI_IMAGE_PATH):
    st.image(FI_IMAGE_PATH, caption='Importância das Features', use_column_width=True)
else:
    st.warning("Arquivo 'feature_importance.png' não encontrado. Por favor, execute o script de treinamento (`save_model.py`) para gerá-lo.")


# --- 4. Interface Interativa para Previsão ---
st.header("🔮 Faça sua Previsão")
st.markdown("Insira os dados para prever a tendência do IBOVESPA para o dia seguinte.")

# Obter os últimos dados do histórico para preencher os defaults
if not historical_df.empty:
    last_day_data = historical_df.iloc[-1]
    default_abertura_ontem = float(last_day_data['abertura'])
    default_ultimo_ontem = float(last_day_data['ultimo'])
    default_abertura_hoje = float(last_day_data['abertura'])
    default_maxima_hoje = float(last_day_data['maxima'])
    default_minima_hoje = float(last_day_data['minima'])
    default_vol_hoje = float(last_day_data['vol_'])
    default_var_hoje = float(last_day_data['var_'])
else:
    default_abertura_ontem = 120000.0
    default_ultimo_ontem = 120000.0
    default_abertura_hoje = 120000.0
    default_maxima_hoje = 121000.0
    default_minima_hoje = 119000.0
    default_vol_hoje = 10000000.0
    default_var_hoje = 0.0

# --- Inputs para as Features ---
st.subheader("Dados do Dia Atual (para prever o dia seguinte)")
col1, col2, col3 = st.columns(3)
with col1:
    input_abertura_hoje = st.number_input(
        "Abertura (Hoje):",
        min_value=0.0,
        value=default_abertura_hoje,
        step=100.0,
        format="%.2f",
        help="Preço de abertura do IBOVESPA para o dia atual."
    )
    input_maxima_hoje = st.number_input(
        "Máxima (Hoje):",
        min_value=0.0,
        value=default_maxima_hoje,
        step=100.0,
        format="%.2f",
        help="Preço máximo do IBOVESPA para o dia atual."
    )
with col2:
    input_minima_hoje = st.number_input(
        "Mínima (Hoje):",
        min_value=0.0,
        value=default_minima_hoje,
        step=100.0,
        format="%.2f",
        help="Preço mínimo do IBOVESPA para o dia atual."
    )
    input_vol_hoje = st.number_input(
        "Volume (Hoje):",
        min_value=0.0,
        value=default_vol_hoje,
        step=100000.0,
        format="%.2f",
        help="Volume de negociações do IBOVESPA para o dia atual."
    )
with col3:
    input_var_hoje = st.number_input(
        "Variação % (Hoje):",
        min_value=-100.0,
        max_value=100.0,
        value=default_var_hoje,
        step=0.01,
        format="%.2f",
        help="Variação percentual do IBOVESPA para o dia atual."
    )

st.subheader("Dados do Dia Anterior (para calcular features lag)")
col4, col5 = st.columns(2)
with col4:
    input_abertura_ontem = st.number_input(
        "Abertura (Ontem):",
        min_value=0.0,
        value=default_abertura_ontem,
        step=100.0,
        format="%.2f",
        help="Preço de abertura do IBOVESPA para o dia anterior."
    )
with col5:
    input_ultimo_ontem = st.number_input(
        "Último (Ontem):",
        min_value=0.0,
        value=default_ultimo_ontem,
        step=100.0,
        format="%.2f",
        help="Preço de fechamento (último) do IBOVESPA para o dia anterior."
    )

# Botão de Previsão
if st.button("Gerar Previsão para o Dia Seguinte"):
    abertura = input_abertura_hoje
    maxima = input_maxima_hoje
    minima = input_minima_hoje
    vol_ = input_vol_hoje
    var_ = input_var_hoje

    abertura_lag1 = input_abertura_ontem
    abertura_fechamento_diff_lag1 = input_abertura_ontem - input_ultimo_ontem

    features_for_prediction = pd.DataFrame([[
        abertura,
        maxima,
        minima,
        vol_,
        var_,
        abertura_lag1,
        abertura_fechamento_diff_lag1
    ]], columns=[
        'abertura',
        'maxima',
        'minima',
        'vol_',
        'var_',
        'abertura_lag1',
        'abertura_fechamento_diff_lag1'
    ])

    try:
        prediction = model.predict(features_for_prediction)[0]
        prediction_proba = model.predict_proba(features_for_prediction)[0]

        st.subheader("Resultado da Previsão")
        if prediction == 1:
            st.success(f"⬆️ **ALTA** (Probabilidade: {prediction_proba[1]*100:.2f}%)")
        else:
            st.error(f"⬇️ **BAIXA** (Probabilidade: {prediction_proba[0]*100:.2f}%)")
        
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'input_abertura_hoje': input_abertura_hoje,
            'input_maxima_hoje': input_maxima_hoje,
            'input_minima_hoje': input_minima_hoje,
            'input_vol_hoje': input_vol_hoje,
            'input_var_hoje': input_var_hoje,
            'input_abertura_ontem': input_abertura_ontem,
            'input_ultimo_ontem': input_ultimo_ontem,
            'prediction': int(prediction),
            'proba_alta': prediction_proba[1],
            'proba_baixa': prediction_proba[0]
        }
        save_log(log_data, 'prediction_log.csv')
        st.caption("Entrada registrada para monitoramento (funcionalidade de log opcional).")

    except Exception as e:
        st.error(f"Erro ao gerar a previsão. Verifique se as features de entrada correspondem ao que o modelo espera ou se o modelo foi carregado corretamente: {e}")
        st.exception(e)

# --- 5. Gráfico Interativo para Análises Temporais e Previsão ---
st.header("📈 Análise Temporal do IBOVESPA (Último Mês)")
st.markdown("Visualize o histórico do IBOVESPA do último mês e o ponto de previsão. O ponto vermelho indica o valor de 'Abertura (Hoje)' que você inseriu.")

if not historical_df.empty:
    last_date_in_data = historical_df['data'].max()
    start_date_for_plot = last_date_in_data - pd.Timedelta(days=30)
    filtered_df = historical_df[historical_df['data'] >= start_date_for_plot].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_df['data'], filtered_df['ultimo'], label='Preço de Fechamento (Último)', color='blue', alpha=0.7)

    next_prediction_date = last_date_in_data + pd.Timedelta(days=1)

    ax.scatter(next_prediction_date, input_abertura_hoje, color='red', s=150, zorder=5, label='Abertura Inserida (Hoje)', marker='X')
    ax.annotate(
        f"Input Abertura: {input_abertura_hoje:.2f}",
        (next_prediction_date, input_abertura_hoje),
        textcoords="offset points",
        xytext=(0,15),
        ha='center',
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=0.5, alpha=0.8)
    )

    ax.set_title('Histórico do IBOVESPA com Ponto de Entrada para Previsão')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend()
    ax.grid(True)
    
    ax.set_ylim(bottom=0, top=200000)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10000))
    ax.ticklabel_format(style='plain', axis='y')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Não foi possível carregar os dados históricos para o gráfico.")


st.sidebar.header("Sobre o Projeto")
st.sidebar.info(
    "Este é o Tech Challenge da Fase 4, desenvolvido para o deploy e monitoramento "
    "de um modelo preditivo do IBOVESPA utilizando Streamlit."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por Alexandre da Silva Oliveira, Carlos Alexandre da Silveira de Souza, Christina Melo Pereira, Daniele dos Santos Ferreira, Marlon Monteiro Militani" )
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Atenção:** Verifique a consistência dos dados de entrada, especialmente os valores de preço (Abertura, Máxima, Mínima, Último), pois inconsistências na escala podem afetar a precisão do modelo.")

