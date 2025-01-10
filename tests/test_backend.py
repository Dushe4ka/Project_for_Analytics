import sys
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory  # Импортируем модуль памяти
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QMenuBar
import os
import faiss  # Убедитесь, что библиотека faiss установлена
import pickle

# Загрузка переменных окружения
load_dotenv()

# Загрузка LLM (Языковой модели)
llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")

# Настройка встраиваний Ollama
embed_model = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")

# Инициализация пустого векторного хранилища
vector_store = None

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

        # Создание меню "Файл"
        file_menu = self.menu_bar.addMenu("Файл")

        # Кнопка для сохранения векторного хранилища
        save_vector_action = file_menu.addAction("Сохранить векторное хранилище")
        save_vector_action.triggered.connect(self.save_vector_store)

        # Кнопка для загрузки векторного хранилища
        load_vector_action = file_menu.addAction("Загрузить векторное хранилище")
        load_vector_action.triggered.connect(self.load_vector_store)

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

        # Создание контейнера и установка центрального виджета
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt);;PDF Files (*.pdf);;All Files (*)")
        if file_name:
            try:
                if file_name.endswith('.pdf'):
                    # Загрузка PDF файла
                    pdf_document = fitz.open(file_name)
                    text = ""
                    for page in pdf_document:
                        text += page.get_text()
                    pdf_document.close()
                    self.text_input.setPlainText(text)
                else:
                    # Загрузка текстового файла
                    with open(file_name, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        self.text_input.setPlainText(file_content)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def save_text(self):
        global vector_store
        user_text = self.text_input.toPlainText()
        if user_text.strip() == "":
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите текст перед сохранением.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunks = text_splitter.split_text(user_text)

        if vector_store is None:
            vector_store = FAISS.from_texts(chunks, embed_model)
        else:
            vector_store.add_texts(chunks)

        QMessageBox.information(self, "Успех", "Текст успешно сохранен!")

    def save_vector_store(self):
        global vector_store
        if vector_store is not None:
            faiss.write_index(vector_store.index, "vector_store.index")
            # Сохраняем docstore и index_to_docstore_id
            with open("docstore.pkl", "wb") as f:
                pickle.dump(vector_store.docstore, f)
            with open("index_to_docstore_id.pkl", "wb") as f:
                pickle.dump(vector_store.index_to_docstore_id, f)

            QMessageBox.information(self, "Успех", "Векторное хранилище успешно сохранено!")
        else:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения.")

    def load_vector_store(self):
        global vector_store
        if os.path.exists("vector_store.index"):
            index = faiss.read_index("vector_store.index")
            # Загружаем docstore и index_to_docstore_id
            with open("docstore.pkl", "rb") as f:
                docstore = pickle.load(f)
            with open("index_to_docstore_id.pkl", "rb") as f:
                index_to_docstore_id = pickle.load(f)

            vector_store = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embed_model)
            QMessageBox.information(self, "Успех", "Векторное хранилище успешно загружено!")
        else:
            QMessageBox.warning(self, "Ошибка", "Файл векторного хранилища не найден.")

    def ask_question(self):
        global vector_store
        user_input = self.input_line.text()

        if not user_input.strip():
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите запрос перед отправкой.")
            return

        try:
            # Сохраняем контекст в памяти
            memory.save_context({"input": user_input}, {"output": ""})

            # Получаем историю сообщений
            chat_history = memory.load_memory_variables({})

            # Формируем контекст из сообщений
            context = chat_history.get("chat_history", "")

            if vector_store is None:
                response = llm.invoke(context)
                self.result_text.setText(response)

                # Сохраняем ответ в памяти
                memory.save_context({"input": user_input}, {"output": response})

                # Обновляем историю
                self.history_text.setPlainText(context)
                return

            retriever = vector_store.as_retriever()
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = retrieval_chain.invoke({"input": context})
            self.result_text.setText(response['answer'])

            # Сохраняем ответ в памяти
            memory.save_context({"input": user_input}, {"output": response['answer']})

            # Обновляем историю
            chat_history = memory.load_memory_variables({})
            self.history_text.setPlainText(chat_history.get("chat_history", ""))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())