import sys
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector  # Импортируем PGVector
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QMenuBar
import os
import psycopg2  # Убедитесь, что библиотека psycopg2 установлена

# Загрузка переменных окружения
load_dotenv()

# Загрузка LLM (Языковой модели)
llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")

# Настройка встраиваний Ollama
embed_model = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")

# Инициализация соединения с базой данных PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

# Инициализация PGVector
vector_store = PGVector(conn, embedding_function=embed_model, table_name="your_vector_table")  # Укажите имя вашей таблицы в PGVector

# Инициализация памяти для хранения истории взаимодействий
memory = ConversationBufferMemory(memory_key="chat_history")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Input for Info Finder")
        self.resize(800, 600)

        # Создание меню
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

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

        # Создание контейнера и установка центрального виджета
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Автоматическая загрузка данных из базы данных при запуске
        self.load_initial_data()

    def load_initial_data(self):
        # Здесь можно реализовать логику загрузки данных из PGVector, если это необходимо
        pass

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

                # Автоматическое сохранение текста в PGVector
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

        # Сохраняем текст в PGVector
        for chunk in chunks:
            vector_store.add_texts([chunk])

        QMessageBox.information(self, "Успех", "Текст успешно сохранен!")

    def ask_question(self):
        user_input = self.input_line.text()

        if not user_input.strip():
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите запрос перед отправкой.")
            return

        try:
            memory.save_context({"input": user_input}, {"output": ""})
            chat_history = memory.load_memory_variables({})
            context = chat_history.get("chat_history", "")

            # Используем PGVector для поиска
            retriever = vector_store.as_retriever()
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = retrieval_chain.invoke({"input": context})
            self.result_text.setText(response['answer'])

            memory.save_context({"input": user_input}, {"output": response['answer']})
            chat_history = memory.load_memory_variables({})
            self.history_text.setPlainText(chat_history.get("chat_history", ""))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())