# Comparação de RMSE entre 30 km/h e 60 km/h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct
from sklearn.linear_model import OrthogonalMatchingPursuit
from matplotlib.ticker import MaxNLocator

# 1. Carregar e Limpar os Dados
df = pd.read_csv('./data/SA_Step_Input_matlab.csv')
df.rename(columns={df.columns[0]: 'Vr'}, inplace=True)
if df['Vr'].dtype == object:
    df['Vr'] = df['Vr'].astype(str).str.replace('%', '').astype(float)

# 2. Separar 500 amostras de cada velocidade
df_30 = df[df['Vr'] <= 30.1].head(500)
df_60 = df[df['Vr'] >= 59.9].head(500)

cols_accz = [f'accz{i}' for i in range(1, 141)]
N = 140

# 3. Configurações da Compressão
percentuais = np.arange(0.20, 0.95, 0.10) # [20%, 30%, ..., 90%] para o gráfico não ficar espremido
Psi = idct(np.eye(N), norm='ortho', axis=0)

# Dicionários para guardar os resultados para o Boxplot e para a Tabela
resultados_rmse = {'30 km/h': [], '60 km/h': []}
dados_tabela = []

# 4. Função auxiliar para processar um dataframe (evita repetir código)
def avaliar_compressao(df_subset, velocidade_label):
    for p in percentuais:
        M = int(p * N)
        rmse_lista = []
        
        np.random.seed(42) # Matriz fixa para a mesma taxa
        indices_amostrados = np.random.choice(N, M, replace=False)
        indices_amostrados.sort()
        Phi = np.eye(N)[indices_amostrados, :]
        Theta = np.dot(Phi, Psi)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, int(M * 0.4)))
        
        for _, row in df_subset.iterrows():
            x_original = row[cols_accz].values.astype(float)
            y = x_original[indices_amostrados]
            
            omp.fit(Theta, y)
            x_reconstruido = np.dot(Psi, omp.coef_)
            
            # Mudança crucial: Usando RMSE em vez de MSE
            rmse = np.sqrt(np.mean((x_original - x_reconstruido)**2))
            rmse_lista.append(rmse)
        
        resultados_rmse[velocidade_label].append(rmse_lista)
        
        # Guardar estatísticas para a tabela do artigo
        dados_tabela.append({
            'Velocidade': velocidade_label,
            'Retenção (%)': int(p * 100),
            'RMSE Média': np.mean(rmse_lista),
            'RMSE Mediana': np.median(rmse_lista),
            'RMSE Desvio Padrão': np.std(rmse_lista),
            'RMSE Max (Outlier)': np.max(rmse_lista)
        })

# Executar a avaliação para as duas velocidades
avaliar_compressao(df_30, '30 km/h')
avaliar_compressao(df_60, '60 km/h')

# 5. Plotagem dos Boxplots Lado a Lado
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
posicoes_x = percentuais * 100

# Boxplot 30 km/h
axs[0].boxplot(resultados_rmse['30 km/h'], positions=posicoes_x, widths=3, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue'), medianprops=dict(color='darkblue', linewidth=2))
axs[0].set_title('Erro de Reconstrução a 30 km/h (500 amostras)', fontweight='bold')
axs[0].set_xlabel('Amostras Retidas (%)')
axs[0].set_ylabel('RMSE (Unidade de Aceleração)')
axs[0].set_xticks(posicoes_x)
axs[0].grid(True, linestyle='--', alpha=0.5)

# Boxplot 60 km/h
axs[1].boxplot(resultados_rmse['60 km/h'], positions=posicoes_x, widths=3, patch_artist=True,
               boxprops=dict(facecolor='salmon', color='darkred'), medianprops=dict(color='darkred', linewidth=2))
axs[1].set_title('Erro de Reconstrução a 60 km/h (500 amostras)', fontweight='bold')
axs[1].set_xlabel('Amostras Retidas (%)')
axs[1].set_xticks(posicoes_x)
axs[1].grid(True, linestyle='--', alpha=0.5)

# Ajuste dos ticks do eixo x para garantir que sejam inteiros
for ax in axs:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticklabels([f"{int(x)}" for x in ax.get_xticks()])

plt.savefig('./results/rmse-error.pdf', dpi=300)
plt.tight_layout()
plt.show()

# 6. Exibir a Tabela de Erros Consolidada para o Artigo
df_tabela = pd.DataFrame(dados_tabela)
df_tabela = df_tabela.round(2) # Arredondar para 2 casas decimais fica mais elegante no artigo
print("\n--- TABELA DE ERROS PARA O ARTIGO ---")
print(df_tabela.to_string(index=False))
