import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_forex_data_from_api(symbol, interval, outputsize, apikey):
    """
    Obtém dados históricos de Forex da API Twelve Data.
    
    A API Twelve Data é usada para obter dados OHLC.
    """
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": apikey
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)
        data = response.json()
        
        if 'values' not in data:
            print(f"Erro ao obter dados da API: {data.get('message', 'Resposta inválida da API')}")
            return None
            
        # Converte a lista de valores para um DataFrame do Pandas
        df = pd.DataFrame(data['values'])
        
        # Renomeia as colunas para o padrão Open, High, Low, Close
        df = df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        
        # Converte as colunas de preço para numérico
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
        
        # Define a coluna 'Date' como índice e converte para datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição HTTP: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

def load_data(file_path):
    """Carrega os dados do arquivo CSV (fallback para teste)."""
    try:
        # Carrega o CSV, definindo a coluna 'Date' como índice e convertendo para datetime
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Garante que as colunas de preço são numéricas
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return None

def identify_patterns(df):
    """
    Identifica os padrões Bullish Engulfing, Bearish Engulfing e Doji.
    """
    # 1. Calcular se a vela é de alta ou de baixa e o tamanho do corpo
    # 1 = Alta (Bullish), -1 = Baixa (Bearish), 0 = Doji/Neutro
    df['Candle_Type'] = df.apply(lambda row: 1 if row['Close'] > row['Open'] else (-1 if row['Close'] < row['Open'] else 0), axis=1)
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    
    # --- Bullish Engulfing (Engolfo de Alta) ---
    # 1. Vela anterior é de baixa (-1)
    # 2. Vela atual é de alta (1)
    # 3. Open atual é menor que o Close anterior
    # 4. Close atual é maior que o Open anterior
    df['Bullish_Engulfing'] = (
        (df['Candle_Type'].shift(1) == -1) & 
        (df['Candle_Type'] == 1) &           
        (df['Open'] < df['Close'].shift(1)) & 
        (df['Close'] > df['Open'].shift(1))   
    )
    
    # --- Bearish Engulfing (Engolfo de Baixa) ---
    # 1. Vela anterior é de alta (1)
    # 2. Vela atual é de baixa (-1)
    # 3. Open atual é maior que o Close anterior
    # 4. Close atual é menor que o Open anterior
    df['Bearish_Engulfing'] = (
        (df['Candle_Type'].shift(1) == 1) &   # Vela anterior é de alta
        (df['Candle_Type'] == -1) &          # Vela atual é de baixa
        (df['Open'] > df['Close'].shift(1)) & # Open atual acima do Close anterior
        (df['Close'] < df['Open'].shift(1))   # Close atual abaixo do Open anterior
    )
    
    # --- Doji ---
    # O corpo da vela é muito pequeno (Open e Close muito próximos).
    # Usaremos um limite arbitrário (ex: 0.0001 para Forex)
    DOJI_THRESHOLD = 0.0001
    df['Doji'] = (df['Body_Size'] <= DOJI_THRESHOLD)
    
    # --- Hammer (Martelo) ---
    # 1. Corpo pequeno (Bullish ou Bearish)
    # 2. Sombra inferior longa (pelo menos 2x o corpo)
    # 3. Sombra superior muito pequena ou inexistente (máximo 10% do corpo)
    # Calcula a sombra inferior (Lower Shadow)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    # Calcula a sombra superior (Upper Shadow)
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Condições para o Martelo
    is_small_body = df['Body_Size'] < (df['High'] - df['Low']) * 0.3 # Corpo é menos de 30% do range total
    is_long_lower_shadow = df['Lower_Shadow'] >= 2 * df['Body_Size']
    is_small_upper_shadow = df['Upper_Shadow'] <= 0.1 * df['Body_Size']
    
    df['Hammer'] = is_small_body & is_long_lower_shadow & is_small_upper_shadow
    
    # --- Shooting Star (Estrela Cadente) ---
    # 1. Corpo pequeno (Bullish ou Bearish)
    # 2. Sombra superior longa (pelo menos 2x o corpo)
    # 3. Sombra inferior muito pequena ou inexistente (máximo 10% do corpo)
    
    # Condições para a Estrela Cadente
    is_long_upper_shadow = df['Upper_Shadow'] >= 2 * df['Body_Size']
    is_small_lower_shadow = df['Lower_Shadow'] <= 0.1 * df['Body_Size']
    
    df['Shooting_Star'] = is_small_body & is_long_upper_shadow & is_small_lower_shadow
    
    # --- Análise de Contexto (Tendência e Pontos Extremos) ---
    
    # 1. Tendência (usando Média Móvel Simples de 20 períodos)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    # Tendência: 1 (Alta), -1 (Baixa), 0 (Lateral)
    df['Trend'] = 0
    df.loc[df['Close'] > df['SMA_20'], 'Trend'] = 1
    df.loc[df['Close'] < df['SMA_20'], 'Trend'] = -1
    
    # 2. Topos e Fundos (usando um método simples de 'pivô')
    # Um ponto é um Topo se for o maior valor em N períodos antes e N períodos depois.
    # Um ponto é um Fundo se for o menor valor em N períodos antes e N períodos depois.
    PIVOT_PERIOD = 5
    
    # Topos
    is_peak = (df['High'] == df['High'].rolling(window=PIVOT_PERIOD*2+1, center=True).max())
    df['Peak'] = is_peak & (df['High'] > df['High'].shift(PIVOT_PERIOD)) # Garante que não é um topo plano
    
    # Fundos
    is_trough = (df['Low'] == df['Low'].rolling(window=PIVOT_PERIOD*2+1, center=True).min())
    df['Trough'] = is_trough & (df['Low'] < df['Low'].shift(PIVOT_PERIOD)) # Garante que não é um fundo plano
    
    # 3. Suporte e Resistência (Identificação simples baseada em topos/fundos recentes)
    # Esta é uma simplificação. A identificação robusta de S/R é complexa.
    # Vamos apenas marcar os topos e fundos como potenciais níveis.
    return df[['Open', 'High', 'Low', 'Close', 'Candle_Type', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Doji', 'Hammer', 'Shooting_Star', 'Trend', 'Peak', 'Trough']]



def run_real_time_monitor(api_key, symbol, interval, output_size, refresh_rate_seconds):
    """
    Executa o monitor de padrões em tempo real (simulado).
    """
    print("--- Monitor de Padrões de Forex Iniciado ---")
    print(f"Par: {symbol} | Intervalo: {interval} | Atualização: {refresh_rate_seconds}s")
    
    while True:
        print(f"\n{'='*50}")
        print(f"Monitorando em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        # 1. Obter dados
        data = get_forex_data_from_api(symbol, interval, output_size, api_key)
        
        if data is None:
            print("\n--- Falha ao obter dados da API. Tentando novamente em breve... ---")
            # Se a API falhar, usa os dados de teste como fallback para não parar o programa
            data = load_data("eurusd_data.csv")
            if data is None:
                time.sleep(refresh_rate_seconds)
                continue
        
        # 2. Identificar padrões
        data_with_patterns = identify_patterns(data)
        
        # 3. Exibir resultados
        patterns_found = False
        
        for pattern_name in ['Bullish_Engulfing', 'Bearish_Engulfing', 'Doji', 'Hammer', 'Shooting_Star', 'Peak', 'Trough']:
            # Filtra apenas as últimas 50 linhas para não poluir muito a saída com dados históricos
            recent_patterns = data_with_patterns.tail(50)
            pattern_dates = recent_patterns[recent_patterns[pattern_name] == True].index.tolist()
            
            if pattern_dates:
                patterns_found = True
                print(f"\n*** Padrão {pattern_name.replace('_', ' ')} ENCONTRADO nas seguintes datas/horas: ***")
                for date in pattern_dates:
                    print(f"- {date.strftime('%Y-%m-%d %H:%M')}")
                
                # Exibir detalhes para padrões de 2 velas (Engulfing)
                if 'Engulfing' in pattern_name:
                    print(f"\n--- Detalhes dos Padrões {pattern_name.replace('_', ' ')} ---")
                    for date in pattern_dates:
                        # Pega a vela atual e a anterior
                        pattern_data = data_with_patterns.loc[[date]].copy()
                        current_loc = data_with_patterns.index.get_loc(date)
                        if current_loc > 0:
                            previous_date = data_with_patterns.index[current_loc - 1]
                            previous_data = data_with_patterns.loc[[previous_date]].copy()
                            
                            print(f"\nData/Hora do Padrão: {date.strftime('%Y-%m-%d %H:%M')}")
                            print(f"Vela Anterior ({previous_date.strftime('%Y-%m-%d %H:%M')}): Open={previous_data['Open'].iloc[0]:.4f}, Close={previous_data['Close'].iloc[0]:.4f}, Type={previous_data['Candle_Type'].iloc[0]}")
                            print(f"Vela Atual ({date.strftime('%Y-%m-%d %H:%M')}): Open={pattern_data['Open'].iloc[0]:.4f}, Close={pattern_data['Close'].iloc[0]:.4f}, Type={pattern_data['Candle_Type'].iloc[0]}")
                
                # Exibir detalhes para padrões de 1 vela (Doji, Hammer, Shooting Star)
                elif pattern_name in ['Doji', 'Hammer', 'Shooting_Star']:
                    print(f"\n--- Detalhes dos Padrões {pattern_name.replace('_', ' ')} ---")
                    for date in pattern_dates:
                        pattern_data = data_with_patterns.loc[[date]].copy()
                        print(f"\nData/Hora do Padrão: {date.strftime('%Y-%m-%d %H:%M')}")
                        print(f"Vela ({date.strftime('%Y-%m-%d %H:%M')}): Open={pattern_data['Open'].iloc[0]:.4f}, Close={pattern_data['Close'].iloc[0]:.4f}, Body Size={pattern_data['Body_Size'].iloc[0]:.5f}, Lower Shadow={pattern_data['Lower_Shadow'].iloc[0]:.5f}, Upper Shadow={pattern_data['Upper_Shadow'].iloc[0]:.5f}")
                
                # Exibir detalhes para Topos e Fundos
                elif pattern_name in ['Peak', 'Trough']:
                    print(f"\n--- Detalhes dos {pattern_name.replace('_', ' ')} ---")
                    for date in pattern_dates:
                        pattern_data = data_with_patterns.loc[[date]].copy()
                        print(f"\nData/Hora do Padrão: {date.strftime('%Y-%m-%d %H:%M')}")
                        print(f"Preço: {pattern_data['Close'].iloc[0]:.4f}, Tendência (SMA_20): {pattern_data['Trend'].iloc[0]}")
        
        if not patterns_found:
            print("\n*** Nenhum dos padrões monitorados foi encontrado nos dados recentes. ***")
            
        print("\n--- Tabela Completa com Resultados Recentes ---")
        print(data_with_patterns.tail(10)[['Open', 'Close', 'Candle_Type', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Doji', 'Hammer', 'Shooting_Star', 'Trend', 'Peak', 'Trough']])
        
        # 4. Esperar antes da próxima execução
        time.sleep(refresh_rate_seconds)

if __name__ == "__main__":
    # --- Configurações ---
    API_KEY = "YOUR_TWELVEDATA_API_KEY" # <-- Insira sua chave de API aqui
    SYMBOL = "EUR/USD"
    INTERVAL = "1min" # Intervalo de tempo para as velas (ex: 1min, 5min, 1h, 1day)
    OUTPUT_SIZE = 100 # Número de velas a serem buscadas
    REFRESH_RATE_SECONDS = 60 # Frequência de atualização em segundos (60s = 1 minuto)
    
    # --- Execução ---
    run_real_time_monitor(API_KEY, SYMBOL, INTERVAL, OUTPUT_SIZE, REFRESH_RATE_SECONDS)
