import soundfile as sf
import numpy as np 

def load_audio(file_path):
    """
    Loads an audio file from the given file path.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing:
            - waveform (numpy.ndarray): The audio data as a NumPy array.
            - sampling_rate (int): The sampling rate of the audio.
        Returns (None, None) if an error occurs (e.g., file not found).
    """
    try:
        waveform, sampling_rate = sf.read(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape of waveform: {waveform.shape}, Sampling rate: {sampling_rate} Hz")
        return waveform, sampling_rate
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    # __file__ is the path to the current script (audio_utils.py)
    # os.path.abspath(__file__) makes it an absolute path
    # os.path.dirname() goes up one level in the directory structure
    
    sample_audio_path = os.path.join(project_root, "data/CommonVoice21.0/cv-corpus-21.0-2025-03-14/de/clips", "common_voice_de_17299268.mp3") 
    # example file path, adjust as needed
    
    print(f"Attempting to load: {sample_audio_path}")
    # Ensure the data folder exists
    data_folder = os.path.join(project_root, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory: {data_folder}")
    else:
        waveform_data, sr = load_audio(sample_audio_path)

        if waveform_data is not None:
            print("Audio loaded successfully in test block!")
        else:
            print("Failed to load audio in test block.")