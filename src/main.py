# Código "main" não está sendo utilizado ainda, mas será usado para organizar o projeto
import os

def setup_project_structure():
    """Cria as pastas necessárias se elas não existirem."""
    folders = ['data', 'results', 'plots']
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Pasta '{folder}' criada com sucesso!")
        else:
            print(f"Pasta '{folder}' já existe.")

if __name__ == "__main__":
    setup_project_structure()
    # Aqui viria seu código de Compressed Sensing...
    print("Iniciando reconstrução do sinal...")