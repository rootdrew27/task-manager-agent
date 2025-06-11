from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLabel,
)
from PySide6.QtCore import QThread
from task_manager_agent import TaskManagerAgent
from capture import Capture
from capture_sd import CaptureSD
from transcriptor import Transcriptor
from dotenv import load_dotenv

INPUT_DEVICE_INDEX = 1
SAMPLE_RATE = 16000
DATA_SIZE = 2  # bytes (int16)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Task Manager Agent")

        # Layout
        self.tasks_display_label = QLabel("🗂️ Current Tasks")
        self.tasks_display_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.tasks_display = QTextEdit()
        self.tasks_display.setReadOnly(True)

        self.request_box_label = QLabel("User Input")
        self.request_box_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.request_box = QTextEdit()
        self.request_box.setMaximumHeight(50)
        self.request_box_button = QPushButton("Submit Text Request")

        self.feedback_display_label = QLabel("Assistant Feedback")
        self.feedback_display_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.feedback_display = QTextEdit()
        self.feedback_display.setReadOnly(True)
        self.feedback_display.setMaximumHeight(50)

        self.button = QPushButton("Enable Mic")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.toggle_mic)

        layout = QVBoxLayout()
        layout.addWidget(self.tasks_display_label)
        layout.addWidget(self.tasks_display)
        layout.addWidget(self.request_box_label)
        layout.addWidget(self.request_box)
        layout.addWidget(self.request_box_button)
        layout.addWidget(self.feedback_display_label)
        layout.addWidget(self.feedback_display)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # # Setup Threads and Workers
        # ## Audio Capture
        self.capture_thread = QThread()
        self.capture_thread.setObjectName("Capture Thread")
        self.capture_worker = CaptureSD(
            input_device_index=INPUT_DEVICE_INDEX, sample_rate=SAMPLE_RATE, chunk_ms=20
        )
        self.capture_worker.moveToThread(self.capture_thread)

        # ## Transcription
        self.transcription_thread = QThread()
        self.transcription_thread.setObjectName("Transcription Thread")
        self.transcription_worker = Transcriptor(
            model_size="base.en", sample_rate=SAMPLE_RATE
        )
        self.transcription_worker.moveToThread(self.transcription_thread)

        # ## Task Manager
        self.task_manager_thread = QThread()
        self.task_manager_thread.setObjectName("Task Manager Thread")
        self.task_manager_worker = TaskManagerAgent()
        self.task_manager_worker.moveToThread(self.task_manager_thread)

        # # Setup Signals and Slots
        # ## Signals from Audio Capture Thread
        self.capture_worker.chunk_ready.connect(self.transcription_worker.transcript)
        # ## Signals from Transcription Thread
        self.transcription_worker.transcript_ready.connect(
            self.task_manager_worker.route_request
        )
        self.transcription_worker.transcript_ready.connect(self.toggle_audio_capture)
        #### TEMP ####
        self.request_box_button.clicked.connect(self.send_text_request)
        #### TEMP ####
 
        # ## Signals from Task Manager Thread
        self.task_manager_worker.agent_done.connect(
            self.display_tasks
        )  # display_tasks handles agent_done events
        self.task_manager_worker.agent_done.connect(self.toggle_audio_capture)
        self.task_manager_worker.need_human_input.connect(self.display_assistant_feedback)
        self.task_manager_worker.need_human_input.connect(self.request_human_input)
        # Start Audio Capture Thread
        self.capture_thread.start()
        # Start Transcription Thread
        self.transcription_thread.start()
        # Start Task Manager Thread
        self.task_manager_thread.start()

        self.display_tasks("")

    ### TEMP ###
    def send_text_request(self):
        text = self.request_box.toPlainText()
        self.transcription_worker.transcript_ready.emit(text)
        self.request_box.clear()

    def closeEvent(self, event):
        self.capture_worker.end_stream.emit()
        self.capture_thread.quit()
        self.capture_thread.wait()

        self.task_manager_thread.quit()
        self.task_manager_thread.wait()

        self.transcription_thread.quit()
        self.transcription_thread.wait()

        event.accept()

    def display_tasks(self, msg: str):
        self.tasks_display.clear()
        for task_name in self.task_manager_worker.task_manager.lut.keys():
            self.tasks_display.append(f"- {task_name}")
        # self.button.setEnabled(True)
        print(f"Message in display_tasks: {msg}")
        self.feedback_display.setText(msg)

    def display_assistant_feedback(self, msg: str):
        self.feedback_display.setText(msg)

    def request_human_input(self, msg):
        self.capture_worker.toggle_capture.emit(True)

    def toggle_mic(self):
        if not self.button.isChecked():
            self.button.setChecked(True)
            self.capture_worker.toggle_capture.emit(True)
        else:
            self.button.setChecked(False)
            self.capture_worker.toggle_capture.emit(False)

    def toggle_audio_capture(self, msg: str, enable: bool = False):
        self.capture_worker.toggle_capture.emit(enable)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.exec()
