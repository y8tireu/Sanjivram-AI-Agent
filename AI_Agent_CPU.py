import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QMessageBox
)
from PyQt5.QtCore import Qt
from transformers import pipeline
import torch


def create_text_generation_pipeline():
    """
    Creates a Hugging Face transformers pipeline for text generation,
    using EleutherAI/gpt-neo-2.7B.

    This model uses 10–12 GB of RAM in float32 on CPU.
    No local file is required it will auto-download from Hugging Faces.
    """
    device_id = -1  # CPU only
    if torch.cuda.is_available():
        # If you had a GPU with enough VRAM, you could do device=0 for faster generation.
        print("GPU is available but this version only supports cpu.")
    print("Loading EleutherAI/gpt-neo-2.7B This may take some time on CPU.")

    generator = pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-2.7B",
        device=device_id
    )
    print("Model loaded successfully.")
    return generator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM (No Local File) - GPT-Neo 2.7B CPU Demo")
        self.resize(800, 600)

        # Create text generation pipeline
        try:
            self.generator = create_text_generation_pipeline()
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            sys.exit(1)

        # UI Setup
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.generate_button = QPushButton("Generate")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        layout.addWidget(QLabel("Ask GPT-Neo 2.7B:"))
        layout.addWidget(self.prompt_input)
        layout.addWidget(self.generate_button)
        layout.addWidget(QLabel("Response:"))
        layout.addWidget(self.output_text)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.generate_button.clicked.connect(self.on_generate)

    def on_generate(self):
        user_prompt = self.prompt_input.text().strip()
        if not user_prompt:
            return

        try:
            # Generate up to 50 new tokens
            result = self.generator(user_prompt, max_new_tokens=50)
            text_out = result[0]["generated_text"]
            self.output_text.append(f"**Prompt**: {user_prompt}")
            self.output_text.append(f"**GPT-Neo 2.7B**: {text_out}\n")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))


def main():
    """
    Run this script in a standard terminal (e.g., `python main.py`) to avoid
    QSocketNotifier warnings. Avoid running it inside Jupyter or IPython without
    proper `%gui qt` setup.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
