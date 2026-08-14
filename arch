import sys
import os
import subprocess
import tempfile
import traceback
from pathlib import Path

import psutil
import sympy as sp

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QPlainTextEdit,
)

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


MODEL_PATH = os.environ.get(
    "ARCHAI_MODEL",
    str(Path.home() / "ArchAI" / "models" / "model.gguf")
)


SYSTEM_PROMPT = r"""
You are ArchAI, a powerful offline Linux AI assistant.

You run locally on Arch Linux.

You are excellent at:
- Python
- C/C++
- Rust
- Bash
- Linux
- system administration
- debugging
- algorithms
- mathematics
- algebra
- calculus
- linear algebra
- statistics
- discrete mathematics
- numerical reasoning
- software architecture

Be precise and practical.

When solving mathematics, show the important reasoning and verify results.

When writing code:
- provide complete working code
- explain important parts
- consider Linux compatibility
- do not invent APIs
- point out bugs clearly

You are an offline assistant. Do not claim to have internet access.
"""


class AIWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, llm, prompt):
        super().__init__()
        self.llm = llm
        self.prompt = prompt

    def run(self):
        try:
            result = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": self.prompt,
                    },
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=4096,
            )

            answer = result["choices"][0]["message"]["content"]
            self.finished.emit(answer)

        except Exception as e:
            self.error.emit(str(e))


class ArchAI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.llm = None
        self.worker = None
        self.current_file = None

        self.setWindowTitle("ArchAI — Offline AI")
        self.resize(1400, 850)

        self.setup_ui()
        self.load_model()

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # ---------------- SIDEBAR ----------------

        sidebar = QWidget()
        sidebar.setFixedWidth(230)

        side = QVBoxLayout(sidebar)
        side.setContentsMargins(15, 15, 15, 15)

        title = QLabel("ARCHAI")
        title.setFont(QFont("Sans", 22, QFont.Bold))

        subtitle = QLabel("OFFLINE INTELLIGENCE")
        subtitle.setObjectName("subtitle")

        side.addWidget(title)
        side.addWidget(subtitle)
        side.addSpacing(25)

        self.status = QLabel("● INITIALIZING")
        self.status.setObjectName("status")

        side.addWidget(self.status)
        side.addSpacing(20)

        self.new_chat_button = QPushButton("＋  New Chat")
        self.open_button = QPushButton("▣  Open File")
        self.save_button = QPushButton("▣  Save File")
        self.math_button = QPushButton("∑  Math")
        self.run_button = QPushButton("▶  Run Python")

        side.addWidget(self.new_chat_button)
        side.addWidget(self.open_button)
        side.addWidget(self.save_button)
        side.addWidget(self.math_button)
        side.addWidget(self.run_button)

        side.addStretch()

        self.system_info = QLabel()
        self.system_info.setObjectName("systemInfo")

        side.addWidget(self.system_info)

        root.addWidget(side)

        # ---------------- MAIN AREA ----------------

        main = QWidget()
        layout = QVBoxLayout(main)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)

        # Chat
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)

        self.chat.setHtml(
            """
            <h2>Welcome to ArchAI</h2>
            <p>Your local AI assistant is starting.</p>
            <p>
            Ask me to write code, debug programs,
            solve mathematics, explain Linux commands,
            or work through a programming problem.
            </p>
            """
        )

        chat_layout.addWidget(self.chat)

        input_row = QHBoxLayout()

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "Ask ArchAI anything..."
        )
        self.input.setMaximumHeight(110)

        self.send_button = QPushButton("SEND")

        input_row.addWidget(self.input)
        input_row.addWidget(self.send_button)

        chat_layout.addLayout(input_row)

        # Editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)

        editor_label = QLabel("CODE / FILE EDITOR")
        editor_label.setObjectName("sectionTitle")

        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Monospace", 11))

        editor_layout.addWidget(editor_label)
        editor_layout.addWidget(self.editor)

        splitter.addWidget(chat_widget)
        splitter.addWidget(editor_widget)

        splitter.setSizes([750, 550])

        layout.addWidget(splitter)

        root.addWidget(main)

        # Events

        self.send_button.clicked.connect(self.ask_ai)
        self.new_chat_button.clicked.connect(self.new_chat)
        self.open_button.clicked.connect(self.open_file)
        self.save_button.clicked.connect(self.save_file)
        self.math_button.clicked.connect(self.math_tool)
        self.run_button.clicked.connect(self.run_python)

        self.update_system_info()

    # ---------------- MODEL ----------------

    def load_model(self):

        if Llama is None:
            self.status.setText("● llama.cpp MISSING")
            self.status.setStyleSheet("color:#ff5555;")
            return

        if not os.path.exists(MODEL_PATH):
            self.status.setText("● MODEL NOT FOUND")
            self.status.setStyleSheet("color:#ff5555;")

            self.chat.append(
                f"""
                <p><b>Model not found.</b></p>
                <p>
                Put a GGUF model at:
                </p>
                <pre>{MODEL_PATH}</pre>
                <p>
                Or set the ARCHAI_MODEL environment variable.
                </p>
                """
            )

            return

        try:

            self.status.setText("● LOADING MODEL")
            self.status.setStyleSheet("color:#ffaa00;")

            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=8192,
                n_threads=max(2, os.cpu_count() or 4),
                n_gpu_layers=-1,
                verbose=False,
            )

            self.status.setText("● ONLINE / LOCAL")
            self.status.setStyleSheet("color:#00ff88;")

            self.chat.append(
                "<p><b>ArchAI is ready.</b> Everything is running locally.</p>"
            )

        except Exception as e:

            self.status.setText("● MODEL ERROR")
            self.status.setStyleSheet("color:#ff5555;")

            self.chat.append(
                f"<p><b>Model error:</b> {e}</p>"
            )

    # ---------------- CHAT ----------------

    def ask_ai(self):

        prompt = self.input.toPlainText().strip()

        if not prompt:
            return

        if self.llm is None:
            QMessageBox.warning(
                self,
                "Model unavailable",
                "Load a GGUF model first."
            )
            return

        self.input.clear()

        self.chat.append(
            f"""
            <p style='color:#4da6ff'>
            <b>You:</b> {prompt}
            </p>
            """
        )

        self.send_button.setEnabled(False)
        self.status.setText("● THINKING")

        self.worker = AIWorker(self.llm, prompt)

        self.worker.finished.connect(self.ai_finished)
        self.worker.error.connect(self.ai_error)

        self.worker.start()

    def ai_finished(self, answer):

        safe = (
            answer
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        self.chat.append(
            f"""
            <p style='color:#00ff88'><b>ArchAI:</b></p>
            <pre>{safe}</pre>
            """
        )

        self.send_button.setEnabled(True)
        self.status.setText("● ONLINE / LOCAL")

    def ai_error(self, error):

        self.chat.append(
            f"""
            <p style='color:#ff5555'>
            <b>ERROR:</b> {error}
            </p>
            """
        )

        self.send_button.setEnabled(True)
        self.status.setText("● ONLINE / LOCAL")

    # ---------------- FILES ----------------

    def open_file(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            str(Path.home()),
            "All Files (*)"
        )

        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.editor.setPlainText(f.read())

            self.current_file = filename

            self.chat.append(
                f"<p>Opened <b>{filename}</b></p>"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Open Error",
                str(e)
            )

    def save_file(self):

        filename = self.current_file

        if not filename:

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save File",
                str(Path.home()),
                "All Files (*)"
            )

        if not filename:
            return

        try:

            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.editor.toPlainText())

            self.current_file = filename

            self.chat.append(
                f"<p>Saved <b>{filename}</b></p>"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Save Error",
                str(e)
            )

    # ---------------- PYTHON RUNNER ----------------

    def run_python(self):

        code = self.editor.toPlainText()

        if not code.strip():
            QMessageBox.warning(
                self,
                "Nothing to run",
                "Put Python code in the editor first."
            )
            return

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False
            ) as f:

                f.write(code)
                filename = f.name

            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(Path.home())
            )

            output = result.stdout

            if result.stderr:
                output += "\n--- STDERR ---\n"
                output += result.stderr

            self.chat.append(
                f"""
                <p><b>Python output:</b></p>
                <pre>{output}</pre>
                """
            )

        except subprocess.TimeoutExpired:
            self.chat.append(
                "<p style='color:#ff5555'>Program timed out after 10 seconds.</p>"
            )

        except Exception as e:

            self.chat.append(
                f"<p style='color:#ff5555'>{e}</p>"
            )

        finally:

            try:
                os.unlink(filename)
            except Exception:
                pass

    # ---------------- MATH ----------------

    def math_tool(self):

        expression, ok = self.get_math_input()

        if not ok:
            return

        try:

            result = sp.sympify(expression)

            simplified = sp.simplify(result)

            expanded = sp.expand(result)

            self.chat.append(
                f"""
                <p style='color:#00ff88'>
                <b>Math Engine</b>
                </p>

                <pre>
Input:
{expression}

Simplified:
{simplified}

Expanded:
{expanded}
                </pre>
                """
            )

        except Exception as e:

            self.chat.append(
                f"""
                <p style='color:#ff5555'>
                Math error: {e}
                </p>
                """
            )

    def get_math_input(self):

        from PySide6.QtWidgets import QInputDialog

        expression, ok = QInputDialog.getText(
            self,
            "Advanced Math",
            "Enter expression:"
        )

        return expression, ok

    # ---------------- CHAT RESET ----------------

    def new_chat(self):

        self.chat.clear()

        self.chat.append(
            """
            <h2>New ArchAI session</h2>
            <p>What are we building?</p>
            """
        )

    # ---------------- SYSTEM INFO ----------------

    def update_system_info(self):

        ram = psutil.virtual_memory()

        self.system_info.setText(
            f"""
CPU: {os.cpu_count()} threads

RAM:
{ram.used // (1024**3)} GB /
{ram.total // (1024**3)} GB

Model:
LOCAL GGUF

Network:
NOT REQUIRED
"""
        )


# ---------------- STYLE ----------------

STYLE = """

QWidget {
    background-color: #0b0e14;
    color: #d8dee9;
    font-family: Sans;
}

QWidget#centralWidget {
    background-color: #0b0e14;
}

QLabel#subtitle {
    color: #6272a4;
    font-size: 10px;
    letter-spacing: 2px;
}

QLabel#status {
    color: #00ff88;
    font-weight: bold;
}

QLabel#sectionTitle {
    color: #4da6ff;
    font-weight: bold;
}

QLabel#systemInfo {
    color: #687080;
    font-family: Monospace;
    font-size: 10px;
}

QPushButton {
    background-color: #151a24;
    border: 1px solid #263143;
    border-radius: 4px;
    padding: 10px;
    color: #cdd6f4;
    text-align: left;
}

QPushButton:hover {
    background-color: #1c2636;
    border-color: #4da6ff;
}

QPushButton:pressed {
    background-color: #24344a;
}

QPlainTextEdit,
QTextEdit {
    background-color: #080b10;
    border: 1px solid #1c2533;
    border-radius: 4px;
    color: #d8dee9;
    selection-background-color: #264f78;
}

QTextEdit {
    padding: 10px;
}

QSplitter::handle {
    background-color: #151a24;
}

"""


def main():

    app = QApplication(sys.argv)

    app.setStyleSheet(STYLE)

    window = ArchAI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
