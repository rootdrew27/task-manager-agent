from faster_whisper import WhisperModel
model = WhisperModel("base.en", device="cuda", compute_type="float32")

if __name__ == "__main__":

    segments, info = model.transcribe("./data/WI007clip.mp3", beam_size=5)
    print("Detected language:", info.language)
    try:
        for segment in list(segments):
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    except Exception as e:
        print("Error:", e)