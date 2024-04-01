
# Neccessary Imports
import os
import pyaudio  # Import the PyAudio library for audio recording
import wave  # Import the wave library for saving the recorded audio in WAV format
import threading  # Import the threading library for managing concurrent operations

class AudioRecorder:
    """
    A class to handle audio recording using the PyAudio library.

    Attributes:
        output_filename (str): The name of the output file where the audio will be saved.
        rate (int): The sample rate of the audio recording.
        chunk (int): The number of audio frames per buffer.
        channels (int): The number of channels in the audio recording.
        format (constant): The sample format of the recording.
        frames (list): A list to hold the recorded audio frames.
        is_recording (bool): A flag to control the recording state.
        p (PyAudio): An instance of the PyAudio class.
        stream: The audio stream created with PyAudio.
    """

    def __init__(self, output_filename = 'output.wav', rate = 44100, chunk = 1024, channels = 2, format = pyaudio.paInt16):
        """
        Initializes the AudioRecorder object with default audio parameters.

        Parameters:
            output_filename (str): The filename for the saved recording.
            rate (int): The audio sample rate.
            chunk (int): The buffer size for the audio stream.
            channels (int): The number of audio channels.
            format: The format of the audio stream.
        """
        self.output_filename = output_filename
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.format = format
        self.frames = []
        self.is_recording = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        """
        Starts the audio recording process in a new thread.
        """
        print("Recording started. Press Enter to stop recording.", flush = True)
        self.is_recording = True
        self.stream = self.p.open(format = self.format,
                                  channels = self.channels,
                                  rate = self.rate,
                                  input = True,
                                  frames_per_buffer = self.chunk)

        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

    def stop_recording(self):
        """
        Stops the audio recording, closes the stream, and saves the audio to a WAV file.
        """
        self.is_recording = False
        print("Stopping recording...", flush=True)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(self.output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        print(f"Recording stopped and saved to {self.output_filename}", flush=True)

