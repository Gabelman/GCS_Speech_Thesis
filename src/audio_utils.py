import soundfile as sf
import numpy as np 
import torch

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


def initialize_vad_model():
    """
    Initializes and returns the Silero VAD model and its utility functions.

    Returns:
        tuple: A tuple containing:
            - model: The loaded VAD model.
            - utils: A dictionary of utility functions from the Silero repo.
    """
    # This downloads the Silero VAD model from PyTorch Hub
    # The model is cached after the first download.
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False) # Set to True to re-download
        print("Silero VAD model initialized successfully.")
        return model, utils
    except Exception as e:
        print(f"Error initializing VAD model: {e}")
        return None, None

def get_speech_timestamps(audio_waveform, vad_model, utils, sampling_rate):
    """
    Uses Silero VAD to find speech timestamps in an audio waveform.

    NOTE: Silero VAD expects audio to be a 16kHz mono torch.Tensor.
          This function currently assumes the input audio is already mono
          and will handle resampling to 16kHz if necessary.

    Args:
        audio_waveform (numpy.ndarray): The audio data.
        vad_model: The initialized Silero VAD model.
        utils (dict): The Silero VAD utility functions.
        sampling_rate (int): The sampling rate of the audio waveform.

    Returns:
        list: A list of dictionaries with 'start' and 'end' times in seconds.
    """
    if vad_model is None or utils is None:
        print("VAD model not initialized.")
        return []

    # Silero VAD works with torch.Tensors
    audio_tensor = torch.from_numpy(audio_waveform).float()

    # VAD expects 16kHz audio. We need to handle resampling if the input is different. (librosa)
    if sampling_rate != 16000:
        try:
            import librosa
            resampler = librosa.resample
            audio_tensor_16k = resampler(y=audio_waveform, orig_sr=sampling_rate, target_sr=16000)
            audio_tensor = torch.from_numpy(audio_tensor_16k).float()
            print(f"Resampled audio from {sampling_rate} Hz to 16000 Hz for VAD.")
        except ImportError:
            print("Error: librosa is not installed. Cannot resample audio to 16kHz for VAD.")
            print("Please install it using: pip install librosa")
            return []
            
    # Get the specific utility function for getting timestamps
    (get_speech_timestamps_func,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Get speech timestamps in samples, then convert to seconds
    try:
        # The VAD model returns timestamps in samples, not seconds.
        speech_timestamps = get_speech_timestamps_func(audio_tensor, vad_model, sampling_rate=16000)
        
        # Let's convert these to seconds for easier use later
        speech_timestamps_seconds = []
        for ts in speech_timestamps:
            speech_timestamps_seconds.append({
                'start': ts['start'] / 16000.0,
                'end': ts['end'] / 16000.0
            })
        
        print(f"VAD found {len(speech_timestamps_seconds)} speech segment(s).")
        return speech_timestamps_seconds
    except Exception as e:
        print(f"Error during VAD processing: {e}")
        return []

# --- Update the Test Block at the bottom of the file ---

if __name__ == "__main__":
    # You'll need to create a 'data' folder in your main project directory
    # and put a sample WAV file there for this test to work.
    # Let's assume you have 'GCS_Speech_Thesis/data/sample.wav'
    
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    sample_audio_path = os.path.join(project_root, "data/CommonVoice21.0/cv-corpus-21.0-2025-03-14/de/clips", "TestSample.mp3") 
    
    print(f"--- Testing audio_utils.py ---")
    print(f"Attempting to load: {sample_audio_path}")

    # Create dummy data folder and check for sample.wav
    data_folder = os.path.join(project_root, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(sample_audio_path):
        print(f"Please place a WAV file at {sample_audio_path} for testing.")
    else:
        # 1. Load the audio
        waveform_data, sr = load_audio(sample_audio_path)

        if waveform_data is not None:
            print("\n--- Testing VAD ---")
            # 2. Initialize the VAD model
            vad_model_instance, vad_utils = initialize_vad_model()

            if vad_model_instance:
                # 3. Get speech timestamps
                timestamps = get_speech_timestamps(waveform_data, vad_model_instance, vad_utils, sr)

                if timestamps:
                    print("\nDetected Speech Segments (in seconds):")
                    for i, ts in enumerate(timestamps):
                        print(f"  Segment {i+1}: Start={ts['start']:.2f}s, End={ts['end']:.2f}s")
                else:
                    print("No speech segments were detected.")
        else:
            print("Failed to load audio, cannot perform VAD test.")