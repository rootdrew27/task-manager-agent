from faster_whisper import WhisperModel
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
import webrtcvad
import logging

# logging.basicConfig(filename='./logs/logs.txt')
# whisper_logger = logging.getLogger("faster_whisper")
# whisper_logger.setLevel(logging.DEBUG)
class Transcriptor(QObject):
    # start_transcriptor = Signal()
    transcript_ready = Signal(str)

    def __init__(self, model_size="tiny.en", sample_rate=16000):
        super().__init__()

        self.sample_rate = sample_rate

        self.model = WhisperModel(model_size, device="cuda", compute_type="float32")
        self.audio_bytes_buffer = []

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

        # self.start_transcriptor.connect(self.transcript)
        self.last_chunk_was_speech = False
        self.speaking = False
        self.max_silence_after_speech = 75 # number of chunks of silence that can pass before speech is transcripted
        self.cur_silence_after_speech = 0

        self.max_speech_sec = 10 # seconds

    def _preprocess(self, audio_bytes: list[bytes]):
        pcm_data = b''.join(audio_bytes)
        return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

    def _transcript(self, audio_bytes: list[bytes]):

        audio_data = self._preprocess(audio_bytes)

        segments, _ = self.model.transcribe(audio_data, language="en", beam_size=5)
         
        # TODO: Add "Hey, Task Manager" check

        text = " ".join([segm.text for segm in segments]).strip()
        # TODO: use regex to filter out shit 
        print(f"Transciption: {text}")
        if len(text) > 4:
            self.transcript_ready.emit(text)
    
    @Slot(bytes)
    def transcript(self, in_data: bytes):
        try:
            is_speech = self.vad.is_speech(in_data, self.sample_rate)

            if self.speaking:
                self.audio_bytes_buffer.append(in_data)
                if not is_speech: # speech has ended
                    self.cur_silence_after_speech += 1
                    if self.cur_silence_after_speech == self.max_silence_after_speech:
                        self._transcript(self.audio_bytes_buffer.copy())
                        self.audio_bytes_buffer.clear()   
                        self.speaking = False
                        self.cur_silence_after_speech = 0

            if is_speech:
                self.speaking = True
                self.audio_bytes_buffer.append(in_data)

        except Exception:
            raise



