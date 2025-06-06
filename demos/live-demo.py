from faster_whisper import WhisperModel
import pyaudio
import numpy as np

model = WhisperModel("tiny.en", device="cuda", compute_type="float32")

# Audio Params
RATE = 16000
CHUNK = 2048
BUFFER = []
BUFFER_DURATION = 3 # seconds

if __name__ == "__main__":

    p = pyaudio.PyAudio()
    stream = p.open(rate=RATE, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=CHUNK, input_device_index=3)

    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # assume that audio data is in a 16-bit PCM format
            audio_chunk = np.frombuffer(buffer=data, dtype=np.int16).astype(np.float32) / 32768.0
            BUFFER.append(audio_chunk)

            if len(BUFFER) >= BUFFER_DURATION * RATE:
                segments, _ = model.transcribe(np.array(BUFFER).flatten(), language="en", beam_size=5, vad_filter=True)
                for seg in segments:
                    print(f"> {seg.text}", flush=True)
                BUFFER.clear()
                
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()