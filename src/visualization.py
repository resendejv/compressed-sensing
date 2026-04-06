import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar e preparar os dados
df = pd.read_csv('./data/SA_Step_Input_matlab.csv')

# Renomeia a primeira coluna para garantir que se chame 'Vr' e limpa caso tenha '%'
df.rename(columns={df.columns[0]: 'Vr'}, inplace=True)
if df['Vr'].dtype == object:
    df['Vr'] = df['Vr'].astype(str).str.replace('%', '').astype(float)

# 2. Sua adaptação inteligente para buscar os índices das velocidades
idx_v30 = df[df['Vr'] <= 30.1].index[0]
idx_v60 = df[df['Vr'] >= 59.9].index[0]

# 3. Criar o eixo X real (de 35° até -35°)
angulos = np.linspace(35, -35, 140)

# Eixos e títulos para o loop
eixos_acc = ['accx', 'accy', 'accz']
titulos = [
    'Aceleração Longitudinal (accx) - Direção do Movimento', 
    'Aceleração Lateral (accy) - Direção do Eixo da Roda', 
    'Aceleração Vertical/Radial (accz) - Compressão do Pneu'
]

# 4. Configurar a figura com 3 subplots empilhados
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 5. Loop para plotar cada eixo
for i, eixo in enumerate(eixos_acc):
    # Selecionar as 140 colunas do eixo atual
    cols_acc = [f'{eixo}{j}' for j in range(1, 141)]
    
    # Extrair os sinais
    sinal_v30 = df.loc[idx_v30, cols_acc].values.astype(float)
    sinal_v60 = df.loc[idx_v60, cols_acc].values.astype(float)
    
    # Plotar no subplot correspondente
    axs[i].plot(angulos, sinal_v30, label=f'30 km/h (Linha {idx_v30})', color='blue', linewidth=2)
    axs[i].plot(angulos, sinal_v60, label=f'60 km/h (Linha {idx_v60})', color='red', linewidth=2)
    
    # Marcação do centro e formatação
    axs[i].axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Centro do Contato (0°)')
    axs[i].set_title(titulos[i], fontweight='bold')
    axs[i].set_ylabel('Aceleração')
    axs[i].grid(True)
    
    # Colocar a legenda apenas no primeiro gráfico para não poluir
    if i == 0:
        axs[i].legend(loc='upper right')

# 6. Formatação final do Eixo X compartilhado
axs[-1].set_xlabel("Ângulo de Rotação (°)", fontsize=12)

# Inverter o eixo X para a ordem cronológica (35 -> -35)
plt.gca().invert_xaxis()

# Ajustar o espaçamento entre os gráficos
plt.tight_layout()
plt.show()