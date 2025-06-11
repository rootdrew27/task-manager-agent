import sounddevice as sd
import numpy as np

DURATION = 1  # seconds
SAMPLE_RATE = 16000
CHANNELS = 32

def callback(indata, frames, time, status):
    print(indata.dtype)
    volume_norm = np.linalg.norm(indata) * 10
    print("|" * int(volume_norm))

print("Recording... Speak into your microphone.")
with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback
    ):
    sd.sleep(DURATION * 1000)
print("Done.")