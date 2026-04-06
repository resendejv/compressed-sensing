import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar os dados
df = pd.read_csv('./data/SA_Step_Input_matlab.csv')

# Tratamento rápido: O Matlab costuma colocar um '%' no primeiro cabeçalho.
# Vamos renomear a primeira coluna para garantir que se chame 'Vr'
df.rename(columns={df.columns[0]: 'Vr'}, inplace=True)
df['Vr'] = df['Vr'].astype(str).str.replace('%', '').astype(float) # Limpa caso o % esteja nos dados

# 2. Criar o eixo X real (de 35° até -35°)
# np.linspace divide o intervalo em 140 pontos iguais
angulos = np.linspace(35, -35, 140)

# 3. Filtrar as linhas pelas velocidades (Vr = 30 e Vr = 60)
# Vamos pegar o primeiro índice (linha) que atenda a cada condição
try:
    idx_v30 = df[df['Vr'] <= 30.1].index[0]
    idx_v60 = df[df['Vr'] >= 59.9].index[0]
except IndexError:
    print("Aviso: Verifique os valores exatos de Vr no seu CSV (ex: 30, 30.0, etc)")

# 4. Selecionar o eixo do acelerômetro que queremos analisar
# Mude para 'accy' ou 'accz' para explorar os outros sinais
eixo_alvo = 'accx' 
cols_acc = [f'{eixo_alvo}{i}' for i in range(1, 141)]

sinal_v30 = df.loc[idx_v30, cols_acc].values.astype(float)
sinal_v60 = df.loc[idx_v60, cols_acc].values.astype(float)

# 5. Plotagem Comparativa
plt.figure(figsize=(12, 6))

# Plotando os sinais
plt.plot(angulos, sinal_v30, label=f'Velocidade = 30 km/h (Linha {idx_v30})', color='blue', linewidth=2)
plt.plot(angulos, sinal_v60, label=f'Velocidade = 60 km/h (Linha {idx_v60})', color='red', linewidth=2)

# Marcação do centro da área de contato
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Centro do Contato (0°)')

# Inverter o eixo X no gráfico para seguir a ordem cronológica da rotação (do 35 para o -35)
plt.gca().invert_xaxis() 

plt.title(f"Comportamento do Sinal ({eixo_alvo.upper()}) na Área de Contato do Pneu")
plt.xlabel("Ângulo de Rotação (°)")
plt.ylabel("Aceleração")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()