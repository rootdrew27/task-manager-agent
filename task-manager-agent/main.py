from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal
from audio_worker import AudioRecorder
from task_manager import TaskManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Task Manager - Voice Agent")

        # Layout
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.button = QPushButton("Start Recording")
        self.button.clicked.connect(self.toggle_recording)

        layout = QVBoxLayout()
        layout.addWidget(self.text_display)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Agent thread
        self.task_manager_thread = QThread()
        self.task_manager_worker = TaskManager()
        self.task_manager_worker.moveToThread(self.task_manager_thread)
        self.task_manager_worker.agent_done.connect(self.display_tasks) # display_tasks handles agent_done events 

        # Audio thread
        self.audio_thread = QThread()
        self.audio_worker = AudioRecorder()
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_worker.transcriptor.transcript_ready.connect(self.disable_recording_button)
        self.audio_worker.transcriptor.transcript_ready.connect(self.task_manager_worker.process_input)
        self.audio_thread.started.connect(self.audio_worker.run)
        self.recording = False

        self.display_tasks()

    def disable_recording_button(self):
        self.button.setDisabled(True)

    def display_tasks(self):
        self.text_display.clear()
        self.text_display.append("🗂️ Current Tasks:")
        for task_name, task_info in self.task_manager_worker.tasks.items():
            self.text_display.append(f"- {task_name}")
        self.button.setDisabled(False)

    def toggle_recording(self):
        if not self.recording:
            self.audio_thread.start()
            self.button.setText("Stop Recording")
        else:
            self.audio_worker.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()
            self.button.setText("Start Recording")
        self.recording = not self.recording

    # def update_text(self, transcript):
    #     self.text_display.append(f"📢 {transcript}")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.resize(400, 300)
    window.show()
    app.exec()
