# inovisao/pytorch-image-classifier-modified/PyTorch-Image-Classifier-Modified-6c1f37d06b3109ac99d998fb9e88472d171d7930/PyTorchRN.py
# --- CONTEÚDO MODIFICADO ---

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import math
import Config_Parametros
import Carregar_Banco
from PIL import Image # Importar a biblioteca Pillow para abrir a imagem

"""
    VERIFICA SE A MÁQUINA ESTÁ USANDO A GPU OU CPU
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")

"""
    TRANSFORMANDO IMAGENS EM TENSORES E NORMALIZANDO
"""
transform = transforms.Compose([
    transforms.Resize((Config_Parametros.tamanho_imagens, Config_Parametros.tamanho_imagens)),
    transforms.ToTensor(),
])

# Carrega o mapeamento de classes uma vez para ser usado globalmente
try:
    full_dataset_for_mapping = datasets.ImageFolder(root=Carregar_Banco.pasta_treino, transform=transform)
    labels_map = {v: k for k, v in full_dataset_for_mapping.class_to_idx.items()}
    total_classes = len(labels_map)
    print(f'Classes encontradas: {labels_map}')
except FileNotFoundError:
    print("AVISO: Diretório do banco de imagens não encontrado. A classificação não funcionará sem o banco de dados.")
    labels_map = {}
    total_classes = 0 # Defina um valor padrão

tamanho_entrada_flatten = Config_Parametros.tamanho_imagens * Config_Parametros.tamanho_imagens * 3


"""
    DEFININDO A CLASSE DA RN
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(tamanho_entrada_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, total_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        output_values = self.linear_relu_stack(x)
        return output_values

"""
    FUNÇÕES DE TREINAMENTO E VALIDAÇÃO
"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), min(batch * dataloader.batch_size, len(dataloader.dataset))
            print(f"Perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    acuracia = correct / size
    print("\n..::: INFORMAÇÕES DE VALIDAÇÃO :::..\n")
    print(f"    Acurácia: {(100*acuracia):>0.1f}% | Perda média: {val_loss:>8f} \n")
    return acuracia

"""
    FUNÇÃO PRINCIPAL DE TREINAMENTO (CHAMADA PELO BOT)
"""
def iniciar_treinamento():
    """
    Função que carrega os dados, inicializa o modelo e executa o ciclo de treinamento e validação.
    """
    print("Iniciando o processo de treinamento...")
    # Carregando o banco de imagens
    training_val_data = datasets.ImageFolder(root=Carregar_Banco.pasta_treino, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(training_val_data))), test_size=Config_Parametros.perc_val)
    training_data = Subset(training_val_data, train_idx)
    val_data = Subset(training_val_data, val_idx)

    # Criando os lotes (batches)
    train_dataloader = DataLoader(training_data, batch_size=Config_Parametros.tamanho_lote, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=Config_Parametros.tamanho_lote, shuffle=True)

    print(f"Total de imagens de treinamento: {len(training_data)}")
    print(f"Total de imagens de validação: {len(val_data)}")

    # Instanciando o modelo
    model = NeuralNetwork().to(device)
    print(model)

    otimizador = torch.optim.Adam(model.parameters(), lr=Config_Parametros.taxa_aprendizagem)
    funcao_perda = nn.CrossEntropyLoss()

    melhor_acuracia = -math.inf
    total_sem_melhora = 0

    # Loop de treinamento
    for t in range(Config_Parametros.epocas):
        print(f"-------------------------------\nÉpoca {t+1}\n-------------------------------")
        train(train_dataloader, model, funcao_perda, otimizador)
        acuracia_val = validation(val_dataloader, model, funcao_perda)

        if acuracia_val > melhor_acuracia:
            print(f">>> Acurácia melhorou ({melhor_acuracia:.3f} --> {acuracia_val:.3f}). Salvando modelo...")
            melhor_acuracia = acuracia_val
            torch.save(model.state_dict(), "modelo_treinado.pth")
            total_sem_melhora = 0
        else:
            total_sem_melhora += 1
            print(f">>> Acurácia não melhorou. Paciência: {total_sem_melhora}/{Config_Parametros.paciencia}")

        if total_sem_melhora >= Config_Parametros.paciencia:
            print(f"\nParada antecipada na época {t+1}!")
            break
    
    print("Terminou a fase de aprendizagem!")
    return "Treinamento concluído com sucesso!"

"""
    FUNÇÃO PARA CLASSIFICAR UMA ÚNICA IMAGEM (CHAMADA PELO BOT)
"""
def classificar_imagem_telegram(image_path):
    """
    Carrega o modelo treinado e classifica uma imagem fornecida.
    """
    if not labels_map:
        return "Erro: As classes não foram carregadas. Verifique o caminho do banco de dados."

    try:
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load("modelo_treinado.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        return "Erro: O arquivo 'modelo_treinado.pth' não foi encontrado. Execute o treinamento primeiro."
    except Exception as e:
        return f"Erro ao carregar o modelo: {e}"

    try:
        # Carrega e transforma a imagem recebida
        imagem = Image.open(image_path).convert("RGB")
        tensor_imagem = transform(imagem).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor_imagem)
            predita_idx = pred[0].argmax(0).item()
            classe_predita = labels_map.get(predita_idx, "Classe desconhecida")
            
            print(f'Imagem recebida: "{image_path}" -> Predição: "{classe_predita}"')
            return f'Predição: "{classe_predita}"'

    except Exception as e:
        return f"Erro ao classificar a imagem: {e}"