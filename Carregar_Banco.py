# inovisao/pytorch-image-classifier-modified/PyTorch-Image-Classifier-Modified-6c1f37d06b3109ac99d998fb9e88472d171d7930/Carregar_Banco.py
# --- CONTEÚDO MODIFICADO ---

import gdown
import zipfile
import os

# --- COLE O LINK COMPARTILHÁVEL DO SEU ARQUIVO .ZIP AQUI ---
# Exemplo: 'https://drive.google.com/uc?id=SEU_ID_AQUI'
DRIVE_FILE_URL = "https://drive.google.com/drive/folders/17WIevxTC0ubQfR7DLpP_KxmLTZsyDe6k?usp=sharing"

# Nome do arquivo zip que será baixado
output_zip_file = "bd_dedos.zip"

# Pasta onde o conteúdo será extraído
pasta_banco_imagens = "./bd_dedos/"

# Função para baixar e extrair o banco de imagens
def preparar_banco_de_imagens():
    """
    Verifica se o banco de imagens já existe, caso contrário,
    baixa do Google Drive e o extrai.
    """
    if os.path.exists(pasta_banco_imagens):
        print(f"O diretório '{pasta_banco_imagens}' já existe. Pulando o download.")
        return

    try:
        print(f"Baixando o banco de imagens de: {DRIVE_FILE_URL}")
        # Baixa o arquivo do Google Drive
        gdown.download(DRIVE_FILE_URL, output_zip_file, quiet=False)
        print("Download concluído com sucesso!")

        print(f"Extraindo '{output_zip_file}' para '{pasta_banco_imagens}'...")
        # Extrai o arquivo .zip
        with zipfile.ZipFile(output_zip_file, 'r') as zip_ref:
            zip_ref.extractall("./")  # Extrai na raiz
        print("Extração concluída!")

        # Opcional: remove o arquivo .zip após a extração
        os.remove(output_zip_file)

    except Exception as e:
        print(f"Ocorreu um erro ao baixar ou extrair o arquivo: {e}")

# Variáveis que serão usadas pelos outros scripts
pasta_raiz = "./"
pasta_treino = os.path.join(pasta_banco_imagens, "train")
pasta_validacao = os.path.join(pasta_banco_imagens, "test")

# Executa a função de preparação ao carregar o módulo
preparar_banco_de_imagens()
print(f"Leitura das imagens configurada para o repositório: {pasta_banco_imagens}")