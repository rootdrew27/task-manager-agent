# Test Audio Levels

import pyaudio
import numpy as np

# Audio Params
RATE = 16000
CHUNK = 2048
BUFFER = []
BUFFER_DURATION = 3 # seconds


p = pyaudio.PyAudio()
stream = p.open(rate=RATE, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=CHUNK, input_device_index=3)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        print(f"Audio level: {np.abs(audio).mean():.2f}")  
except KeyboardInterrupt:
    print("Stopped")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()