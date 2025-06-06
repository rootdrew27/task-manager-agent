from PySide6.QtCore import QObject, Signal, Slot
import pyaudio

class Capture(QObject):
    
    toggle_capture = Signal(bool)
    end_stream = Signal()
    chunk_ready = Signal(bytes)

    def __init__(self, input_device_index: int, sample_rate: int = 16000, chunk_ms: int = 20):
        """A QObject for capturing and forwarding audio input.

        Args:
            input_device_index (int, optional): The index of the input device. View devices index with input_device_scan.py
            sample_rate (int, optional): The sample rate of audio input. Defaults to 16000.
            chunk_ms (int, optional): The timeframe of each audio chunk, in milliseconds . Defaults to 20.
        """
        super().__init__()

        self.pya = pyaudio.PyAudio()
        self.input_device_index = input_device_index
        self.device_info = self.pya.get_device_info_by_index(input_device_index)
        channels= self.device_info["maxInputChannels"] # NOTE: if the input device does not provide samples from all channels (the max) then the frames_per_buffer calculation will be wrong
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate  
        self.frames_per_buffer = int(sample_rate * chunk_ms / 1000)
        self.running = False

        assert isinstance(channels, int)
        assert chunk_ms >= 10 and chunk_ms <= 30

        self.channels = channels

        self.stream = self.pya.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            start=False,
            frames_per_buffer=self.frames_per_buffer,
            input_device_index=self.input_device_index,
            stream_callback=self._process_audio # type: ignore
        )

        self.toggle_capture.connect(self._toggle_capture)
        self.end_stream.connect(self.end)

        self.count = 0

    # TODO: move this callback to a different thread??
    def _process_audio(self, in_data: bytes, frame_count: int, time_info: dict, status_flag: int) -> tuple[None, int]:
        # print(f"in_data type: {type(in_data)}, frame_count type: {type(frame_count)}, time_info type: {type(time_info)}, status_flag type: {type(status_flag)}\n")

        # self.count += 1
        # if self.count >= 100:
        #     print(f"Capturing info: {type(in_data)}")
        #     print(f"Length of in_data: {len(in_data)}") # in number of bytes
        #     self.count = 0

        self.chunk_ready.emit(in_data)

        return (None, pyaudio.paContinue)

    @Slot(bool)
    def _toggle_capture(self, enable: bool):
        if enable:
            if not self.running:
                print("Starting Stream!")
                self.running = True
                self.stream.start_stream()
            else:
                print("Already running!")
        else: # disable
            if self.running:
                print("🛑 Stopping stream")
                self.stream.stop_stream()
                self.running = False
            else:
                print("Already stopped!")


    @Slot()
    def end(self):
        if self.stream:
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
            self.pya.terminate()

            