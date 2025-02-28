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
    QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QMenuBar, QComboBox
)
import logging
import os
import uuid

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Загрузка LLM (Языковой модели)
llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")

# Настройка встраиваний Ollama
embed_model = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")

# Инициализация клиента Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)
collection_name = "documents"

# Инициализация памяти для хранения истории взаимодействий
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Список чатов
chat_sessions = {}


def initialize_qdrant():
    try:
        # Проверка доступности сервера Qdrant через список коллекций
        collections = qdrant_client.get_collections()
        logger.info("Соединение с Qdrant успешно установлено.")

        # Определение размера вектора
        test_vector = embed_model.embed_query("test")
        vector_size = len(test_vector)

        # Проверка существования коллекции
        if collection_name not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Коллекция '{collection_name}' успешно создана.")
        else:
            logger.info(f"Коллекция '{collection_name}' уже существует.")

    except Exception as e:
        logger.error(f"Ошибка при инициализации Qdrant: {e}")


initialize_qdrant()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Input for Info Finder")
        self.resize(800, 600)

        # Инициализация памяти (можно сделать это также в функции load_data)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Создание меню
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Создание меню "Файл"
        file_menu = self.menu_bar.addMenu("Файл")

        # Кнопка для сохранения векторного хранилища
        save_vector_action = file_menu.addAction("Сохранить векторное хранилище")
        save_vector_action.triggered.connect(self.save_vector_store)

        self.layout = QVBoxLayout()

        # Заголовок
        self.label = QLabel("Введите текст для анализа:")
        self.layout.addWidget(self.label)

        # Поле ввода текста
        self.text_input = QTextEdit()
        self.layout.addWidget(self.text_input)

        # Кнопка загрузки файла
        self.load_button = QPushButton("Загрузить файл")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        # Кнопка сохранения текста
        self.save_button = QPushButton("Сохранить текст")
        self.save_button.clicked.connect(self.save_text)
        self.layout.addWidget(self.save_button)

        # Заголовок для запроса
        self.label_query = QLabel("Введите запрос:")
        self.layout.addWidget(self.label_query)

        # Поле ввода запроса
        self.input_line = QLineEdit()
        self.layout.addWidget(self.input_line)

        # Кнопка для запроса
        self.query_button = QPushButton("Задать вопрос")
        self.query_button.clicked.connect(self.ask_question)
        self.layout.addWidget(self.query_button)

        # Поле для отображения результата
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        # Поле для отображения истории запросов
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.layout.addWidget(QLabel("История запросов:"))
        self.layout.addWidget(self.history_text)

        # Дропдаун для выбора чата
        self.chat_selector = QComboBox()
        self.chat_selector.addItem("Новый чат")
        self.layout.addWidget(QLabel("Выберите чат:"))
        self.layout.addWidget(self.chat_selector)

        # Кнопка для создания нового чата
        self.new_chat_button = QPushButton("Новый чат")
        self.new_chat_button.clicked.connect(self.create_new_chat)
        self.layout.addWidget(self.new_chat_button)

        # Создание контейнера и установка центрального виджета
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "Text Files (*.txt);;PDF Files (*.pdf);;Word Files (*.doc *.docx);;All Files (*)"
        )
        if file_name:
            try:
                if file_name.endswith('.pdf'):
                    pdf_document = fitz.open(file_name)
                    text = ""
                    for page in pdf_document:
                        text += page.get_text()
                    pdf_document.close()
                    self.text_input.setPlainText(text)
                elif file_name.endswith(('.doc', '.docx')):
                    from docx import Document
                    doc = Document(file_name)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    self.text_input.setPlainText(text)
                else:
                    with open(file_name, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        self.text_input.setPlainText(file_content)

                # Автоматическое сохранение текста в Pinecone
                self.save_text()

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def save_text(self):
        user_text = self.text_input.toPlainText()
        if user_text.strip() == "":
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите текст перед сохранением.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(user_text)

        try:
            for chunk in chunks:
                embedding = embed_model.embed_query(chunk)
                logger.debug(f"Вектор для текста: {chunk[:30]}...: {embedding}")
                qdrant_client.upsert(collection_name=collection_name, points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {"text": chunk}
                }])
            QMessageBox.information(self, "Успех", "Текст успешно сохранен в Qdrant!")
        except Exception as e:
            logger.error(f"Ошибка сохранения текста в Qdrant: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить текст: {str(e)}")

    def save_vector_store(self):
        QMessageBox.information(self, "Информация", "Данные сохраняются автоматически в Qdrant.")

    def create_new_chat(self):
        new_chat_id = str(uuid.uuid4())
        chat_sessions[new_chat_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_selector.addItem(f"Чат {new_chat_id[:8]}")
        QMessageBox.information(self, "Новый чат", f"Чат с ID {new_chat_id[:8]} был создан.")

    def ask_question(self):
        user_input = self.input_line.text()

        if not user_input.strip():
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите запрос перед отправкой.")
            return

        try:
            # Инициализация памяти для текущего чата
            if not hasattr(self, 'memory'):
                self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Сохраняем контекст в памяти
            self.memory.save_context({"input": user_input}, {"output": ""})

            # Получаем историю сообщений
            chat_history = self.memory.load_memory_variables({})

            # Формируем контекст из сообщений
            context = chat_history.get("chat_history", "")

            # Генерация вектора запроса с использованием модели встраивания
            query_embedding = embed_model.embed_query(user_input)

            # Поиск ближайших соседей в Qdrant
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=5,  # Ограничиваем количество результатов
                score_threshold=0.5  # Исключаем нерелевантные результаты
            )

            # Если результаты поиска пустые, генерируем ответ ИИ
            if not search_results:
                # Если нет данных в Qdrant, генерируем ответ с использованием LLaMA
                response = llm(user_input)  # Генерация ответа на основе запроса
                self.result_text.setPlainText(str(response))  # Передаем строку в setPlainText
            else:
                # Если есть релевантные данные, передаем их в LLaMA с контекстом
                relevant_text = ['Представь что ты преподаватель, на основе представленных данных далее дай максимально раскрытый вариант ответа без повторов']
                for res in search_results:
                    # Проверяем, что "text" является строкой
                    text = res.payload.get("text", "")
                    if isinstance(text, list):
                        # Если это список, объединяем его в строку
                        relevant_text.append("\n".join(str(item) for item in text))
                    else:
                        relevant_text.append(str(text))  # Преобразуем в строку, если это не список

                # Собираем все найденные данные в одну строку
                combined_text = "\n".join(relevant_text)

                # Генерация ответа с учетом найденных данных
                context = combined_text + "\n\n" + user_input  # Формируем контекст для LLaMA
                response = llm(context)  # Генерация ответа с использованием контекста
                self.result_text.setPlainText(str(response))  # Передаем строку в setPlainText

            # Сохраняем ответ в памяти
            self.memory.save_context({"input": user_input}, {"output": response})

            # Обновляем историю
            chat_history = self.memory.load_memory_variables({})
            # Преобразуем историю в строку, если это список
            history = "\n".join(str(item) for item in chat_history.get("chat_history", []))
            self.history_text.setPlainText(history)

        except Exception as e:
            logger.error(f"Ошибка запроса Qdrant: {e}")
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
