
# Neccessary Imports
import os
import pyaudio  # Import the PyAudio library for audio recording
import wave  # Import the wave library for saving the recorded audio in WAV format
import threading  # Import the threading library for managing concurrent operations

class AudioRecorder:
    """
    A class to handle audio recording using the PyAudio library.

    Attributes:
        filename (str): The name of the output file where the audio will be saved.
        rate (int): The sample rate of the audio recording.
        chunk (int): The number of audio frames per buffer.
        channels (int): The number of channels in the audio recording.
        audio_format (constant): The sample format of the recording.
        frames (list): A list to hold the recorded audio frames.
        audio (PyAudio): An instance of the PyAudio class.
        stream: The audio stream created with PyAudio.
    """

    def __init__(self, filename='output.wav', rate=44100, chunk=1024, channels=1, format=pyaudio.paInt16):
        """
        Initializes the AudioRecorder object with default audio parameters.

        Parameters:
            filename (str): The filename for the saved recording.
            rate (int): The audio sample rate.
            chunk (int): The buffer size for the audio stream.
            channels (int): The number of audio channels.
            format: The format of the audio stream.
        """
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.audio_format = format
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.audio_format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        self.frames = []

    def record(self, duration):
        """
        Records for a fixed duration of time (in seconds).
        """
        print("Recording...")
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = self.stream.read(self.chunk)
            self.frames.append(data)
        print("Finished recording.")

    def save(self):
        """
        Closes the stream, and saves the audio to a WAV file.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Save the audio as .WAV
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Audio saved as {self.filename}")