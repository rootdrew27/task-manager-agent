from PySide6.QtCore import QObject
import pyaudio
import webrtcvad
from sounddevice import CallbackFlags
import numpy as np
import queue

from transcriptor import Transcriptor

class AudioRecorder(QObject):

    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.channels = 1
        self.frame_duration = 20  # milliseconds
        self.frame_samples = int(self.sample_rate * self.frame_duration / 1000)  # 480
        self.frame_bytes = self.frame_samples * 2  # 16-bit PCM → 2 bytes/sample
        self.device_index = 3
        self.running = True

        self.audio_interface = pyaudio.PyAudio()
        self.transcriptor = Transcriptor("base.en")

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)

        self.speech_buffer = []
        self.silence_counter = 0
        self.max_silence_frames = 3

    def _callback(self, in_data, frame_count, time_info, status):
        if status != 0:
            print("⚠️ PyAudio status:", status)

        # webrtcvad requires raw 16-bit mono PCM, 10/20/30ms length
        if len(in_data) != self.frame_bytes:
            print(f"❌ Frame is wrong size: {len(in_data)} bytes (expected {self.frame_bytes})")
            return (None, pyaudio.paContinue)

        is_speech = self.vad.is_speech(in_data, self.sample_rate)

        if is_speech:
            self.speech_buffer.append(in_data)
            self.silence_counter = 0
        else:
            self.silence_counter += 1
            if self.speech_buffer and self.silence_counter > self.max_silence_frames:
                print("Silence detected — transcribing...")
                pcm_data = b''.join(self.speech_buffer)
                audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                self.transcriptor.transcript(audio_np)
                self.speech_buffer.clear()
                self.silence_counter = 0

        return (None, pyaudio.paContinue)

    def run(self):
        print("🎙️ Starting PyAudio stream...")
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_samples,
            stream_callback=self._callback,
            input_device_index=self.device_index,
        )

        stream.start_stream()

        try:
            while self.running:
                pass
        finally:
            print("🛑 Closing stream...")
            stream.stop_stream()
            stream.close()
            self.audio_interface.terminate()

    def stop(self):
        print("🛑 Stopping audio...")
        self.running = False
