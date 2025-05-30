from faster_whisper import WhisperModel
import numpy as np
from PySide6.QtCore import QObject, Signal

class Transcriptor(QObject):
    transcript_ready = Signal(str)

    def __init__(self, model_size="base.en", sample_rate=16000, min_duration_sec=2.0, overlap_sec=0.5):

        super().__init__()

        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration_sec)
        self.overlap_samples = int(sample_rate * overlap_sec)

        self.model = WhisperModel(model_size, device="cuda", compute_type="float32")

        # Buffer to hold leftover overlap from last chunk
        self.overlap_buffer = np.empty((0,), dtype=np.float32)

    def transcript(self, audio):
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
        for segm in segments:
            print(segm.text)
            self.transcript_ready.emit(segm.text)
    