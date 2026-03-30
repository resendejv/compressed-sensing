import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft

# Lê o arquivo usando Pandas
df = pd.read_csv("./data/SA_Step_Input_matlab.csv", comment='%',header=None)
primeiras_cols = ['Vr', 'Fy', 'SA']
num_total_aceleracoes = df.shape[1] - len(primeiras_cols)
num_acc_cols_por_eixo = 140


# Cria as listas de nomes (accx0 a accx140, accy0 a accy140, etc.)
accx_names = [f'accx{i}' for i in range(num_acc_cols_por_eixo)]
accy_names = [f'accy{i}' for i in range(num_acc_cols_por_eixo)]
accz_names = [f'accz{i}' for i in range(num_acc_cols_por_eixo)]

# A lista final de nomes de colunas
col_names = primeiras_cols + accx_names + accy_names + accz_names

# Corta a lista de nomes para o tamanho real do DataFrame, se necessário
df.columns = col_names[:df.shape[1]]

print(f"Dimensões (Amostras, Variáveis): {df.shape}")
print(f"Colunas de aceleração identificadas por eixo: {num_acc_cols_por_eixo}")
print(df.columns[[0, 1, 2, 3, 4, -2, -1]].tolist()) # Mostra as primeiras e últimas para checagem
print(df.columns.tolist())

# --- Seleção e Criação do Eixo de Tempo ---
# Sinal alvo: accx1 (o primeiro sensor de aceleração)   
sinal_alvo = df['accx1'].values
N = len(sinal_alvo)
Fs = 1000
tempo = np.arange(N) / Fs

# --- Plotagem no Domínio do Tempo (Análise da Resposta Degrau) ---
plt.figure(figsize=(12, 10))

# Plotagem do Sinal no Tempo
plt.subplot(2, 1, 1)
plt.plot(tempo, sinal_alvo)
plt.title(f'Sinal de Aceleração (accx1) - Teste Step Input ({N} Amostras)')
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração')
plt.grid(True)

# --- Análise de Esparsidade: Transformada Rápida de Fourier (FFT) ---

# Calcula a FFT (Transformada de Fourier)
sinal_fft = scipy.fft.fft(sinal_alvo)
sinal_fft_abs = np.abs(sinal_fft)

# Frequências: só precisamos da primeira metade (Frequência de Nyquist)
frequencias = scipy.fft.fftfreq(N, 1/Fs)[:N//2]
magnitudes = sinal_fft_abs[:N//2] / N # Normaliza a magnitude

# Plotagem do Sinal no Domínio da Frequência
plt.subplot(2, 1, 2)
# Focamos nas 1000 primeiras frequências para visualizar melhor a esparsidade
plt.plot(frequencias[:1000], magnitudes[:1000])
plt.title('Espectro de Magnitude da FFT (Domínio de Esparsidade Candidato)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude Normalizada')
plt.grid(True)
plt.tight_layout()
plt.show()