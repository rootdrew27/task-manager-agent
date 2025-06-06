import pyaudio

p = pyaudio.PyAudio()

device_count = p.get_device_count()

for i in range(device_count):
    device_info = p.get_device_info_by_index(i)
    if device_info['maxInputChannels'] == 0:
        continue
    print(f"Device Index: {i}")
    print(f"Device Name: {device_info['name']}")
    print(f"Max Input Channels: {device_info['maxInputChannels']}")
    print(f"Max Output Channels: {device_info['maxOutputChannels']}")
    print(f"Default Sample Rate: {device_info['defaultSampleRate']}")
    print("-" * 20)