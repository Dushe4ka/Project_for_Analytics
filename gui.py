from PyQt6.QtWidgets import QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog
from PyQt6.QtGui import QFont
from model import OllamaModel
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Input for Info Finder")
        self.resize(800, 600)
        self.layout = QVBoxLayout()
        self.setStyleSheet("background-color: #f5f5f5;")

        # Заголовок
        self.label = QLabel("Введите текст для анализа:")
        font = QFont("Arial", 16)
        font.setBold(True)
        self.label.setFont(font)
        self.layout.addWidget(self.label)

        # Поле ввода текста
        self.text_input = QTextEdit()
        self.text_input.setFont(QFont("Arial", 12))
        self.text_input.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc; padding: 10px;")
        self.layout.addWidget(self.text_input)

        # Кнопки
        self.load_button = QPushButton("📁 Загрузить файл")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; border: none; margin-top: 10px;")
        self.layout.addWidget(self.load_button)

        self.save_button = QPushButton("💾 Сохранить текст")
        self.save_button.clicked.connect(self.save_text)
        self.save_button.setFont(QFont("Arial", 12))
        self.save_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border: none; margin-top: 10px;")
        self.layout.addWidget(self.save_button)

        # Заголовок для запроса
        self.label_query = QLabel("Введите запрос:")
        font_query = QFont("Arial", 16)
        font_query.setBold(True)
        self.label_query.setFont(font_query)
        self.layout.addWidget(self.label_query)

        # Поле ввода запроса
        self.input_line = QLineEdit()
        self.input_line.setFont(QFont("Arial", 12))
        self.input_line.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc; padding: 10px;")
        self.layout.addWidget(self.input_line)

        # Кнопка для запроса
        self.query_button = QPushButton("❓ Задать вопрос")
        self.query_button.clicked.connect(self.ask_question)
        self.query_button.setFont(QFont("Arial", 12))
        self.query_button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; border: none; margin-top: 10px;")
        self.layout.addWidget(self.query_button)

        # Поле для отображения результата
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Arial", 12))
        self.result_text.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc; padding: 10px; margin-top: 10px;")
        self.layout.addWidget(self.result_text)

        # Поле для отображения истории запросов
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Arial", 12))
        self.history_text.setStyleSheet("background-color: #ffffff; border: 1px solid #cccccc; padding: 10px; margin-top: 10px;")
        self.layout.addWidget(QLabel("История запросов:"))
        self.layout.addWidget(self.history_text)

        # Создание контейнера и установка центрального виджета
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Инициализация модели
        self.model = OllamaModel()

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    self.text_input.setPlainText(file_content)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def save_text(self):
        user_text = self.text_input.toPlainText()
        if user_text.strip() == "":
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите текст перед сохранением.")
            return

        self.model.save_text(user_text)
        QMessageBox.information(self, "Успех", "Текст успешно сохранен!")

    def ask_question(self):
        user_input = self.input_line.text()
        if not user_input.strip():
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, введите запрос перед отправкой.")
            return

        try:
            response = self.model.ask_question(user_input)
            self.result_text.setText(response)
            self.history_text.append(f"<div style='background-color: #D1E7DD; padding: 5px;'>User: {user_input}</div>")
            self.history_text.append(f"<div style='background-color: #CFE2F3; padding: 5px;'>Ollama: {response}</div>")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")

