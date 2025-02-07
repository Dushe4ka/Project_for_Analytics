import sys
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QMenuBar
)
import logging
import os
import uuid
import cv2
import ctypes

try:
    libc = ctypes.CDLL("C:\\Windows\\System32\\msvcrt.dll")
    print("Библиотека msvcrt.dll загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки libc: {e}")
from PIL import Image, UnidentifiedImageError
import pytesseract
import magic
from moviepy.editor import VideoFileClip
import numpy as np
from datetime import datetime
from io import BytesIO
import base64
import subprocess

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


# Проверка зависимостей
def check_dependencies():
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Tesseract OCR не установлен или не добавлен в PATH")
        QMessageBox.critical(None, "Ошибка", "Tesseract OCR не установлен или не добавлен в PATH")

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg не установлен или не добавлен в PATH")
        QMessageBox.critical(None, "Ошибка", "FFmpeg не установлен или не добавлен в PATH")


check_dependencies()

# Настройка путей
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')

# Инициализация моделей
try:
    llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")
    embed_model = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")
    whisper_model = whisper.load_model("base")
except Exception as e:
    logger.error(f"Ошибка инициализации моделей: {e}")
    QMessageBox.critical(None, "Ошибка", f"Не удалось инициализировать модели: {str(e)}")

# Инициализация клиента Qdrant
try:
    qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
    collection_name = "multimedia_documents"
except Exception as e:
    logger.error(f"Ошибка подключения к Qdrant: {e}")
    QMessageBox.critical(None, "Ошибка", f"Не удалось подключиться к Qdrant: {str(e)}")


def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        test_vector = embed_model.embed_query("test")
        vector_size = len(test_vector)

        if collection_name not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Коллекция '{collection_name}' создана")
        else:
            logger.info(f"Коллекция '{collection_name}' уже существует")

    except Exception as e:
        logger.error(f"Ошибка инициализации Qdrant: {e}")
        QMessageBox.critical(None, "Ошибка", f"Ошибка инициализации Qdrant: {str(e)}")


initialize_qdrant()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Processor")
        self.resize(1000, 800)
        self.setup_ui()
        self.mime = magic.Magic()
        self.whisper_model = whisper_model

    def setup_ui(self):
        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu("Файл")
        save_vector_action = file_menu.addAction("Сохранить в Qdrant")
        save_vector_action.triggered.connect(self.save_vector_store)

        layout = QVBoxLayout()

        # Элементы интерфейса
        self.file_label = QLabel("Выберите файл для обработки:")
        self.load_button = QPushButton("Загрузить файл")
        self.load_button.clicked.connect(self.load_file)

        self.progress_label = QLabel("Готов к работе")
        self.content_preview = QTextEdit()
        self.content_preview.setReadOnly(True)

        self.query_input = QLineEdit()
        self.ask_button = QPushButton("Задать вопрос")
        self.ask_button.clicked.connect(self.ask_question)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # Добавление элементов в layout
        layout.addWidget(self.file_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.content_preview)
        layout.addWidget(QLabel("Введите запрос:"))
        layout.addWidget(self.query_input)
        layout.addWidget(self.ask_button)
        layout.addWidget(QLabel("Результаты:"))
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "",
            "Все файлы (*);;"
            "Текст (*.txt *.pdf *.doc *.docx);;"
            "Изображения (*.png *.jpg *.jpeg);;"
            "Аудио (*.mp3 *.wav);;"
            "Видео (*.mp4 *.avi)"
        )
        if file_name:
            try:
                file_type = self.mime.from_file(file_name)
                self.progress_label.setText("Обработка файла...")
                QApplication.processEvents()

                if 'image' in file_type:
                    self.process_image(file_name)
                elif 'audio' in file_type:
                    self.process_audio(file_name)
                elif 'video' in file_type:
                    self.process_video(file_name)
                else:
                    self.process_text(file_name)

                self.progress_label.setText("Обработка завершена")
                QMessageBox.information(self, "Успех", "Файл успешно обработан")

            except Exception as e:
                self.progress_label.setText("Ошибка обработки")
                logger.error(f"Ошибка обработки файла: {str(e)}")
                QMessageBox.critical(self, "Ошибка", f"Ошибка обработки файла: {str(e)}")

    def process_text(self, file_path):
        try:
            text = ""
            if file_path.endswith('.pdf'):
                with fitz.open(file_path) as doc:
                    text = "".join([page.get_text() for page in doc])
            elif file_path.endswith(('.doc', '.docx')):
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            self.content_preview.setPlainText(text[:5000] + "\n... [ПРЕДПРОСМОТР ОГРАНИЧЕН]")
            self.save_to_qdrant(text, "text")

        except Exception as e:
            logger.error(f"Ошибка обработки текста: {str(e)}")
            raise Exception(f"Ошибка обработки текстового файла: {str(e)}")

    def process_image(self, file_path):
        try:
            img = Image.open(file_path)
            try:
                text = pytesseract.image_to_string(img, lang='rus+eng')
            except Exception as e:
                logger.warning(f"OCR ошибка: {str(e)}")
                text = ""

            try:
                description = self.describe_image(img)
            except Exception as e:
                logger.warning(f"Ошибка описания изображения: {str(e)}")
                description = ""

            content = f"Текст с изображения:\n{text}\n\nОписание изображения:\n{description}"
            self.content_preview.setPlainText(content[:5000] + "\n... [ПРЕДПРОСМОТР ОГРАНИЧЕН]")
            self.save_to_qdrant(content, "image")

        except (UnidentifiedImageError, IOError) as e:
            raise Exception(f"Неподдерживаемый формат изображения: {str(e)}")
        except Exception as e:
            raise Exception(f"Ошибка обработки изображения: {str(e)}")

    def process_audio(self, file_path):
        try:
            result = self.whisper_model.transcribe(file_path)
            text = result["text"]
            self.content_preview.setPlainText(text[:5000] + "\n... [ПРЕДПРОСМОТР ОГРАНИЧЕН]")
            self.save_to_qdrant(text, "audio")

        except whisper.WhisperError as e:
            raise Exception(f"Ошибка транскрибации аудио: {str(e)}")
        except Exception as e:
            raise Exception(f"Ошибка обработки аудио: {str(e)}")

    def process_video(self, file_path):
        try:
            # Обработка аудио
            try:
                video = VideoFileClip(file_path)
                audio_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_path)
                audio_text = self.process_audio(audio_path)
                os.remove(audio_path)
            except Exception as e:
                audio_text = ""
                logger.warning(f"Ошибка обработки аудио из видео: {str(e)}")

            # Обработка кадров
            cap = cv2.VideoCapture(file_path)
            frame_text = ""
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 30 == 0:
                    try:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_text += pytesseract.image_to_string(img) + "\n"
                    except Exception as e:
                        logger.warning(f"Ошибка обработки кадра: {str(e)}")

                frame_count += 1

            cap.release()
            content = f"Аудио транскрипт:\n{audio_text}\n\nТекст с кадров:\n{frame_text}"
            self.content_preview.setPlainText(content[:5000] + "\n... [ПРЕДПРОСМОТР ОГРАНИЧЕН]")
            self.save_to_qdrant(content, "video")

        except Exception as e:
            raise Exception(f"Ошибка обработки видео: {str(e)}")

    def describe_image(self, img):
        try:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            response = llm.generate(
                [f"Опиши это изображение подробно на русском языке:\n![image](data:image/jpeg;base64,{img_str})"])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"Ошибка генерации описания: {str(e)}")
            return "Не удалось сгенерировать описание"

    def save_to_qdrant(self, content, content_type):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            chunks = text_splitter.split_text(content)

            for chunk in chunks:
                embedding = embed_model.embed_query(chunk)
                qdrant_client.upsert(collection_name=collection_name, points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "type": content_type,
                        "timestamp": datetime.now().isoformat()
                    }
                }])

        except Exception as e:
            logger.error(f"Ошибка сохранения в Qdrant: {str(e)}")
            raise Exception(f"Ошибка сохранения данных: {str(e)}")

    def save_vector_store(self):
        QMessageBox.information(self, "Информация", "Данные сохраняются автоматически в Qdrant")

    def ask_question(self):
        query = self.query_input.text()
        if not query:
            QMessageBox.warning(self, "Предупреждение", "Введите запрос")
            return

        try:
            embedding = embed_model.embed_query(query)
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=3
            )

            response = "Результаты поиска:\n\n" + "\n\n".join(
                f"[{res.payload['type']}] {res.payload['text']}"
                for res in results
            )

            self.result_display.setPlainText(response)

        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка выполнения запроса: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        QMessageBox.critical(None, "Ошибка", f"Приложение завершено с ошибкой: {str(e)}")