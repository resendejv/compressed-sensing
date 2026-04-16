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

# 2. Selecionar um conjunto de dados para a estatística
# Pegando todas as linhas com velocidade ao redor de 60 km/h. 
df_subset = df[df['Vr'] <= 30.1].head(500)
cols_accz = [f'accz{i}' for i in range(1, 141)]
N = 140 # Tamanho do sinal

# 3. Configurar os percentuais e as estruturas de armazenamento
percentuais = np.arange(0.2, 1.0, 0.1) # 20% a 90%
mse_por_taxa = [] # Vai armazenar uma lista de MSEs para cada taxa de compressão
sinais_reconstruidos = {} # Para plotar um exemplo visual no 2º gráfico

# Pegamos a primeira linha do subset apenas para servir de exemplo visual no gráfico da direita
ref_idx = df_subset.index[100] 
x_ref_original = df_subset.loc[ref_idx, cols_accz].values.astype(float)

# Matriz de Esparsidade Psi (DCT Inversa)
Psi = idct(np.eye(N), norm='ortho', axis=0)

# 4. Loop de Reconstrução em Múltiplas Linhas
for p in percentuais:
    M = int(p * N) # Número de medições
    mses_desta_taxa = [] # Guarda os erros de todas as linhas para a taxa 'p'
    
    # Criamos a matriz de medição UMA VEZ para esta taxa (simula o hardware físico)
    np.random.seed(42) 
    indices_amostrados = np.random.choice(N, M, replace=False)
    indices_amostrados.sort()
    
    Phi = np.eye(N)[indices_amostrados, :]
    Theta = np.dot(Phi, Psi)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, int(M * 0.4)))
    
    # Iterando sobre cada linha (rotação do pneu) do nosso subset
    for idx, row in df_subset.iterrows():
        x_original = row[cols_accz].values.astype(float)
        
        # Subamostragem
        y = x_original[indices_amostrados]
        
        # Reconstrução
        omp.fit(Theta, y)
        s_reconstruido = omp.coef_
        x_reconstruido = np.dot(Psi, s_reconstruido)
        
        # Cálculo do MSE para esta linha específica
        mse = np.mean((x_original - x_reconstruido)**2)
        mses_desta_taxa.append(mse)
        
        # Se for a nossa linha de referência, salvamos para o gráfico visual
        if idx == ref_idx and round(p, 2) in [0.3, 0.5, 0.8]:
            sinais_reconstruidos[round(p, 2)] = x_reconstruido
            
    # Adiciona a lista de erros desta taxa na lista principal do Boxplot
    mse_por_taxa.append(mses_desta_taxa)

# 5. Plotagem dos Resultados
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Boxplot da Variância do Erro
# Multiplicamos as posições por 100 para ficar no eixo X como porcentagem
posicoes_x = percentuais * 100 
axs[0].boxplot(mse_por_taxa, positions=posicoes_x, widths=2, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))

axs[0].set_title('Distribuição do Erro (MSE) vs. Taxa de Amostragem', fontweight='bold')
axs[0].set_xlabel('Amostras Retidas do Sinal Original (%)')
axs[0].set_ylabel('Erro Quadrático Médio (MSE)')
axs[0].set_xticks(posicoes_x)
axs[0].set_xticklabels([f"{int(x)}%" for x in posicoes_x])
axs[0].grid(True, linestyle='--', alpha=0.5)

# Gráfico 2: Comparação Visual da Reconstrução (Apenas para a linha de referência)
angulos = np.linspace(35, -35, 140)
axs[1].plot(angulos, x_ref_original, label='Original (100%)', color='black', linewidth=2, zorder=5)

cores = {0.3: 'red', 0.5: 'orange', 0.8: 'green'}
for p_chave, x_rec in sinais_reconstruidos.items():
    axs[1].plot(angulos, x_rec, label=f'Reconstruído ({int(p_chave*100)}%)', 
                color=cores[p_chave], linestyle='dashed', alpha=0.8)

axs[1].invert_xaxis()
axs[1].set_title(f'Visualização da Reconstrução (Linha Ref: {ref_idx})', fontweight='bold')
axs[1].set_xlabel('Ângulo de Rotação (°)')
axs[1].set_ylabel('Aceleração Radial (accz)')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()