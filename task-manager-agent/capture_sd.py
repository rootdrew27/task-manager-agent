from PySide6.QtCore import QObject, Signal, Slot
import sounddevice as sd

class CaptureSD(QObject):
    toggle_capture = Signal(bool)
    end_stream = Signal()
    chunk_ready = Signal(bytes)

    def __init__(self, input_device_index: int, sample_rate: int = 16000, chunk_ms: int = 20):
        """A QObject for capturing and forwarding audio input.

        Args:
            input_device_index (int): The index of the input device.
            sample_rate (int, optional): The sample rate of audio input. Defaults to 16000.
            chunk_ms (int, optional): The timeframe of each audio chunk, in milliseconds. Defaults to 20.
        """
        super().__init__()

        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms

        # Query device info to get channel count
        # device_info = sd.query_devices(self.input_device_index, 'input')
        # assert isinstance(device_info, dict)
        self.channels = 1 # device_info['max_input_channels']

        assert isinstance(self.channels, int)
        assert chunk_ms >= 10 and chunk_ms <= 30

        self.frames_per_buffer = int(self.sample_rate * self.chunk_ms / 1000)
        self.running = False

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frames_per_buffer,
            channels=self.channels,
            dtype='int16',
            device=self.input_device_index,
            callback=self._process_audio
        )

        self.toggle_capture.connect(self._toggle_capture)
        self.end_stream.connect(self.end)

    def _process_audio(self, indata, frames, time, status):
        """
        sounddevice provides indata as a numpy array, shape = (frames, channels).
        Convert to bytes (little-endian, int16) and emit.
        """
        # Optionally, flatten in case channels=1 (gets shape (N,))
        data_bytes = indata.tobytes()
        self.chunk_ready.emit(data_bytes)

    @Slot(bool)
    def _toggle_capture(self, enable: bool):
        print(f"_toggle_capture called with enable=({enable})")
        if enable:
            if not self.running:
                print("Starting Stream!")
                self.stream.start()
                self.running = True
            else:
                print("Already running!")
        else:
            if self.running:
                print("🛑 Stopping stream")
                self.stream.stop()
                self.running = False
            else:
                print("Already stopped!")

    @Slot()
    def end(self):
        if self.stream:
            if self.running:
                self.stream.stop()
                self.running = False
            self.stream.close()
