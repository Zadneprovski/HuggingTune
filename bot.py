from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Загрузка модели и токенизатора
MODEL_PATH = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def predict(text: str):
    """Функция для предсказания класса текста"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Положительный отзыв" if prediction == 1 else "Отрицательный отзыв"


# Функции бота
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет! Отправьте мне текст, и я предскажу его тональность.")


def handle_message(update: Update, context: CallbackContext):
    text = update.message.text
    sentiment = predict(text)
    update.message.reply_text(f"Анализ тональности: {sentiment}")


def main():
    TOKEN = ""
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()