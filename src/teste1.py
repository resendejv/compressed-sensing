import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.linear_model import OrthogonalMatchingPursuit

# 1. Carregar os dados
df = pd.read_csv('./data/SA_Step_Input_matlab.csv')

# Pegando a PRIMEIRA linha e extraindo apenas os 140 pontos do eixo Z (accz)
accz_cols = [f'accz{i}' for i in range(1, 141)]
x_original = df.loc[0, accz_cols].values.astype(float)
N = len(x_original) # N = 140

# 2. Criar a subamostragem (Matriz de Medição Phi)
# Vamos amostrar apenas 50% do sinal original
M = int(0.5 * N)

# Escolhendo índices aleatórios para as "medições"
np.random.seed(101) # Para reprodutibilidade
indices_amostrados = np.random.choice(N, M, replace=False)
indices_amostrados.sort()

# Criando o vetor de medições (y)
y = x_original[indices_amostrados]

# Matriz de medição Phi (M x N) - Identidade com linhas deletadas
Phi = np.eye(N)[indices_amostrados, :]

# 3. Base de Esparsidade (Psi) - Usando DCT
# A matriz Psi converte do domínio da frequência para o tempo
Psi = idct(np.eye(N), norm='ortho', axis=0)

# A matriz Theta é a combinação da Medição com a Base de Esparsidade (Theta = Phi * Psi)
Theta = np.dot(Phi, Psi)

# 4. Reconstrução usando OMP (Orthogonal Matching Pursuit)
# Queremos encontrar o vetor esparso 's' tal que y = Theta * s
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=int(M/2)) # Estimativa de coeficientes não nulos
omp.fit(Theta, y)
s_reconstruido = omp.coef_

# 5. Voltar para o domínio do tempo
z_reconstruido = np.dot(Psi, s_reconstruido)

# 6. Visualização dos Resultados
plt.figure(figsize=(10, 6))
plt.plot(x_original, label='Sinal Original (140 pontos)', color='blue', linewidth=2)
plt.scatter(indices_amostrados, y, color='red', label=f'Medições ({M} pontos)', zorder=5)
plt.plot(z_reconstruido, label=f'Sinal Reconstruído via CS ({int((M/N)*100)}%)', color='green', linestyle='dashed', linewidth=2)
plt.title("Compressed Sensing no Sinal de Aceleração (Eixo Z) do Pneu")
plt.xlabel("Índice da Amostra")
plt.ylabel("Aceleração")
plt.legend()
plt.grid(True)

plt.rcParams.update({'font.size': 14})
plt.savefig('./results/reconstruction-z.pdf', dpi=300)
plt.show()

# Cálculo do Erro
erro_mse = np.mean((x_original - z_reconstruido)**2)
print(f"Erro Quadrático Médio (MSE) da Reconstrução: {erro_mse:.4f}")