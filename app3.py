import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
@st.cache_resource # <--- COMENTE ESTA LINHA TEMPORARIAMENTE PARA DEBUGAR O CACHE
def load_model(model_path):
    """Carrega o modelo treinado."""
    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.exists(model_path):
        st.error(f"Erro: O arquivo do modelo '{model_path}' não foi encontrado. "
                 "Certifique-se de que o arquivo 'forest.joblib' está no mesmo diretório do 'app.py'.")
        st.stop() # Interrompe a execução do Streamlit
        return None # Retorna None para indicar falha

    try:
        # Tenta carregar o objeto do arquivo
        raw_loaded_object = joblib.load(model_path)
        
        # --- DEBUGGING CRÍTICO: Mostra o tipo do objeto carregado ---
        # st.write(f"DEBUG: Tipo do objeto carregado por joblib.load('{model_path}'): **{type(raw_loaded_object)}**")
        
        # Verifica se o objeto carregado é uma string (o que não deveria ser para um modelo)
        if isinstance(raw_loaded_object, str):
            st.error(f"Erro: O arquivo '{model_path}' foi carregado como uma **string** em vez de um modelo. "
                     "Isso geralmente indica que o arquivo está corrompido ou não foi salvo corretamente "
                     "como um modelo joblib. Por favor, verifique se você salvou seu modelo usando "
                     "`joblib.dump(seu_modelo, 'forest.joblib')` e se o arquivo não foi alterado.")
            st.stop()
            return None
        
        # Se chegou aqui, assumimos que é um modelo válido
        model = raw_loaded_object
        st.success(f"Modelo '{model_path}' carregado com sucesso. Tipo esperado: {type(model)}")
        return model
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo '{model_path}': {type(e).__name__}: {e}. "
                 "Verifique a integridade do arquivo 'forest.joblib'.")
        st.stop()
        return None

@st.cache_data # Cacheia o carregamento dos dados para performance
def load_historical_data(file_path):
    """Carrega e pré-processa os dados históricos do IBOVESPA."""
    try:
        df = pd.read_csv(file_path, sep=',') # Ajuste sep para ',' conforme sua correção
        
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

model = load_model(MODEL_PATH)
historical_df = load_historical_data(DATA_PATH)

# --- REMOVENDO A LINHA REDUNDANTE, POIS JÁ TEMOS O DEBUG DENTRO DE load_model ---
# st.write(f"Tipo da variável 'model' global após load_model: {type(model)}")

if model is None or historical_df is None:
    st.error("A aplicação não pode iniciar devido a erros no carregamento do modelo ou dados históricos. Por favor, verifique os logs acima.")
    st.stop()

# --- 3. Painel de Métricas de Validação do Modelo ---
st.header("📊 Performance do Modelo")
st.info(f"**Modelo Utilizado:** Random Forest Classifier")
st.metric(label="Acurácia de Validação", value="80.00%") # Acurácia reportada por você

st.markdown("""
    > *Lembre-se: Esta acurácia foi obtida durante a fase de treinamento e validação em um conjunto de teste 
    apresentado no Tech Challenge 02.*
""")

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
        # Agora, se 'model' for None (devido a um erro no carregamento), esta linha vai falhar
        # e o 'st.exception(e)' vai capturar e mostrar o erro de 'NoneType'
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
        st.exception(e) # Para depuração

# --- 5. Gráfico Interativo para Análises Temporais e Previsão ---
st.header("📈 Análise Temporal do IBOVESPA")
st.markdown("Visualize o histórico do IBOVESPA e o ponto de previsão. O ponto vermelho indica o valor de 'Abertura (Hoje)' que você inseriu.")

if not historical_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_df['data'], historical_df['ultimo'], label='Preço de Fechamento (Último)', color='blue', alpha=0.7)

    last_historical_date = historical_df['data'].max()
    next_prediction_date = last_historical_date + pd.Timedelta(days=1)

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