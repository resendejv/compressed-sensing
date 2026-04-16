# Código para reconstrução de sinais usando DCT, OMP e análise de erro.
# Comparando diferentes percentuais de amostragem.
# O sinal accz é o mais promissor para reconstrução.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct
from sklearn.linear_model import OrthogonalMatchingPursuit

# 1. Carregar os dados
df = pd.read_csv('./data/SA_Step_Input_matlab.csv')
df.rename(columns={df.columns[0]: 'Vr'}, inplace=True)
if df['Vr'].dtype == object:
    df['Vr'] = df['Vr'].astype(str).str.replace('%', '').astype(float)

# Pegar uma linha a 60 km/h
idx_v60 = df[df['Vr'] >= 59.9].index[0]

# Selecionar o sinal accz (Aceleração Radial)
cols_accz = [f'accz{i}' for i in range(1, 141)]
x_original = df.loc[idx_v60, cols_accz].values.astype(float)
N = len(x_original) # 140 amostras

# 2. Configurar os percentuais de amostragem (20% a 90%)
percentuais = np.arange(0.20, 0.95, 0.1) # [0.20, 0.30 ... 0.90]
erros_mse = []
sinais_reconstruidos = {} # Vamos guardar alguns para plotar depois

# Matriz de Esparsidade Psi (DCT Inversa)
Psi = idct(np.eye(N), norm='ortho', axis=0)

# 3. Loop de Reconstrução e Cálculo de Erro
np.random.seed(42) # Fixar a semente para resultados consistentes

for p in percentuais:
    M = int(p * N) # Número de medições
    
    # Escolher índices aleatórios para subamostragem
    indices_amostrados = np.random.choice(N, M, replace=False)
    indices_amostrados.sort()
    
    y = x_original[indices_amostrados]
    Phi = np.eye(N)[indices_amostrados, :]
    Theta = np.dot(Phi, Psi)
    
    # OMP para reconstruir (assumimos que a esparsidade é proporcional a M)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, int(M * 0.4)))
    omp.fit(Theta, y)
    s_reconstruido = omp.coef_
    
    # Voltar para o domínio do tempo
    x_reconstruido = np.dot(Psi, s_reconstruido)
    
    # Calcular e armazenar o erro
    mse = np.mean((x_original - x_reconstruido)**2)
    erros_mse.append(mse)
    
    # Salvar os sinais de 30%, 50% e 80% para visualização
    if round(p, 2) in [0.3, 0.5, 0.8]:
        sinais_reconstruidos[round(p, 2)] = x_reconstruido

# 4. Plotagem dos Resultados
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Curva de Erro
axs[0].plot(percentuais * 100, erros_mse, marker='o', color='purple', linewidth=2, markersize=8)
axs[0].set_title('Erro de Reconstrução (MSE) vs. Taxa de Amostragem', fontweight='bold')
axs[0].set_xlabel('Amostras Retidas do Sinal Original (%)')
axs[0].set_ylabel('Erro Quadrático Médio (MSE)')
axs[0].grid(True, linestyle='--', alpha=0.7)

# Gráfico 2: Comparação Visual da Reconstrução
angulos = np.linspace(35, -35, 140)
axs[1].plot(angulos, x_original, label='Original (100%)', color='black', linewidth=2, zorder=5)

cores = {0.3: 'red', 0.5: 'orange', 0.8: 'green'}
for p_chave, x_rec in sinais_reconstruidos.items():
    axs[1].plot(angulos, x_rec, label=f'Reconstruído ({int(p_chave*100)}%)', 
                color=cores[p_chave], linestyle='dashed', alpha=0.8)

axs[1].invert_xaxis()
axs[1].set_title('Impacto Visual da Reconstrução (Sinal accz)', fontweight='bold')
axs[1].set_xlabel('Ângulo de Rotação (°)')
axs[1].set_ylabel('Aceleração')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
print(erros_mse)