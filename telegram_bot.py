# telegram_bot.py

import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import PyTorchRN  # Importa seu script modificado

# Configure o logging para ver erros
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- COLOQUE SEU TOKEN AQUI ---
TELEGRAM_BOT_TOKEN = "7538688960:AAGJjkFnHu0AMLUeGDG6DmhzUVSD68abFgo" 

# Função para o comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Olá! Sou um bot de classificação de imagens.\n\n"
             "Envie o comando /treinar para iniciar o aprendizado da rede neural.\n\n"
             "Ou me envie uma imagem para que eu possa classificá-la!"
    )

# Função para o comando /treinar
async def treinar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="Iniciando o treinamento... Isso pode levar vários minutos. Por favor, aguarde.")
    
    try:
        # Chama a função de treinamento do seu script
        resultado = PyTorchRN.iniciar_treinamento()
        await context.bot.send_message(chat_id=chat_id, text=resultado)
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"Ocorreu um erro durante o treinamento: {e}")

# Função para lidar com imagens recebidas
async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        # Pega a foto de melhor resolução
        photo_file = await update.message.photo[-1].get_file()
        
        # Cria uma pasta para salvar as imagens temporariamente, se não existir
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
            
        file_path = os.path.join('downloads', f'{photo_file.file_id}.jpg')
        
        # Baixa a imagem
        await photo_file.download_to_drive(file_path)
        
        await context.bot.send_message(chat_id=chat_id, text="Imagem recebida! Classificando...")

        # Chama a função de classificação
        prediction = PyTorchRN.classificar_imagem_telegram(file_path)
        
        # Envia o resultado de volta
        await context.bot.send_message(chat_id=chat_id, text=prediction)

        # Opcional: apaga a imagem após a classificação
        os.remove(file_path)

    except Exception as e:
        logging.error(f"Erro ao processar a imagem: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"Ocorreu um erro ao processar sua imagem: {e}")


if __name__ == '__main__':
    if TELEGRAM_BOT_TOKEN == "SEU_TOKEN_AQUI":
        print("ERRO: Por favor, adicione o token do seu bot no arquivo telegram_bot.py")
    else:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Adiciona os handlers
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('treinar', treinar_command))
        application.add_handler(MessageHandler(filters.PHOTO, image_handler))
        
        print("Bot iniciado! Pressione Ctrl+C para parar.")
        
        # Inicia o bot
        application.run_polling()