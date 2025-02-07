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
    QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QMenuBar,
    QSplitter, QListWidget, QListWidgetItem, QInputDialog, QHBoxLayout, QMenu
)
from PyQt6.QtCore import Qt
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

# Хранилище чатов
chat_sessions = {}


def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections()
        logger.info("Соединение с Qdrant успешно установлено.")

        test_vector = embed_model.embed_query("test")
        vector_size = len(test_vector)

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
        self.setWindowTitle("AI Chat")
        self.resize(900, 600)

        # Создание главного контейнера
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # **Левая панель (список чатов)**
        self.chat_list = QListWidget()
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self.show_chat_context_menu)
        self.chat_list.itemDoubleClicked.connect(self.rename_chat)
        self.chat_list.itemClicked.connect(self.switch_chat)

        # **Правая панель (основной интерфейс)**
        self.central_widget = QWidget()
        chat_layout = QVBoxLayout()

        # Поле загрузки файлов
        self.load_button = QPushButton("Загрузить файл")
        self.load_button.clicked.connect(self.load_file)
        chat_layout.addWidget(self.load_button)

        # Поле ввода запроса
        self.input_line = QLineEdit()
        chat_layout.addWidget(QLabel("Введите запрос:"))
        chat_layout.addWidget(self.input_line)

        # Кнопка для запроса
        self.query_button = QPushButton("Отправить")
        self.query_button.clicked.connect(self.ask_question)
        chat_layout.addWidget(self.query_button)

        # Поле вывода результата
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        chat_layout.addWidget(self.result_text)

        # Поле истории чата
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        chat_layout.addWidget(QLabel("История сообщений:"))
        chat_layout.addWidget(self.history_text)

        # Кнопка создания нового чата
        self.new_chat_button = QPushButton("Новый чат")
        self.new_chat_button.clicked.connect(self.create_new_chat)
        chat_layout.addWidget(self.new_chat_button)

        self.central_widget.setLayout(chat_layout)

        # Добавление в сплиттер
        splitter.addWidget(self.chat_list)
        splitter.addWidget(self.central_widget)
        splitter.setSizes([200, 700])  # Устанавливаем размеры колонок

        main_layout.addWidget(splitter)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Инициализация первого чата
        self.create_new_chat()

    def create_new_chat(self):
        chat_id = str(uuid.uuid4())[:8]
        chat_name = f"Чат {len(chat_sessions) + 1}"
        chat_sessions[chat_id] = {
            "name": chat_name,
            "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        }
        item = QListWidgetItem(chat_name)
        item.setData(Qt.ItemDataRole.UserRole, chat_id)
        self.chat_list.addItem(item)
        self.chat_list.setCurrentItem(item)
        self.switch_chat(item)

    def switch_chat(self, item):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        chat_data = chat_sessions[chat_id]
        self.setWindowTitle(f"AI Chat - {chat_data['name']}")

        self.current_chat_id = chat_id
        self.memory = chat_data["memory"]

        chat_history = self.memory.load_memory_variables({})
        history = "\n".join(str(item) for item in chat_history.get("chat_history", []))
        self.history_text.setPlainText(history)

    def rename_chat(self, item):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        new_name, ok = QInputDialog.getText(self, "Переименование чата", "Введите новое название:", text=item.text())

        if ok and new_name.strip():
            chat_sessions[chat_id]["name"] = new_name.strip()
            item.setText(new_name.strip())

    def show_chat_context_menu(self, pos):
        item = self.chat_list.itemAt(pos)
        if item:
            menu = QMenu(self)
            delete_action = menu.addAction("Удалить чат")
            action = menu.exec(self.chat_list.mapToGlobal(pos))
            if action == delete_action:
                self.delete_chat(item)

    def delete_chat(self, item):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        del chat_sessions[chat_id]
        self.chat_list.takeItem(self.chat_list.row(item))

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
                elif file_name.endswith(('.doc', '.docx')):
                    from docx import Document
                    doc = Document(file_name)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                else:
                    with open(file_name, 'r', encoding='utf-8') as f:
                        text = f.read()

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
                chunks = text_splitter.split_text(text)

                for chunk in chunks:
                    embedding = embed_model.embed_query(chunk)
                    qdrant_client.upsert(collection_name=collection_name, points=[{
                        "id": str(uuid.uuid4()), "vector": embedding, "payload": {"text": chunk}
                    }])

                QMessageBox.information(self, "Успех", "Файл загружен в Qdrant!")

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

            # Получаем контекст чата (история сообщений)
            context = chat_history.get("chat_history", "")

            # Если context является списком, объединим его в строку
            if isinstance(context, list):
                context = "\n".join(str(item) for item in context)

            # Проверяем, содержит ли контекст информацию, связанную с текущим запросом
            if any(keyword.lower() in context.lower() for keyword in user_input.split()):
                # Если контекст связан с запросом, используем его для ответа
                response = llm(context + "\n\n" + user_input)
                self.result_text.setPlainText(str(response))  # Передаем строку в setPlainText
            else:
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
                    response = llm(user_input)  # Генерация ответа на основе запроса
                    self.result_text.setPlainText(str(response))  # Передаем строку в setPlainText
                else:
                    # Если есть релевантные данные, передаем их в LLaMA с контекстом
                    relevant_text = [
                        'Представь что ты преподаватель, на основе представленных данных далее дай максимально раскрытый вариант ответа без повторов']
                    for res in search_results:
                        # Проверяем, что "text" является строкой
                        text = res.payload.get("text", "")
                        if isinstance(text, list):
                            relevant_text.append("\n".join(str(item) for item in text))
                        else:
                            relevant_text.append(str(text))  # Преобразуем в строку, если это не список

                    # Собираем все найденные данные в одну строку
                    combined_text = "\n".join(relevant_text)

                    # Генерация ответа с учетом найденных данных
                    context = combined_text + "\n\n" + user_input  # Формируем контекст для LLaMA
                    response = llm.invoke(context)  # Генерация ответа с использованием контекста
                    self.result_text.setPlainText(str(response))  # Передаем строку в setPlainText

            # Сохраняем ответ в памяти
            self.memory.save_context({"input": user_input}, {"output": response})

            # Обновляем историю
            chat_history = self.memory.load_memory_variables({})
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
