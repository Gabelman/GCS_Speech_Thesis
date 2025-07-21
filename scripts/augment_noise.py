import os
import soundfile as sf
import numpy as np
import random
import librosa 

def calculate_rms(audio):
    """Calculates the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def add_noise(clean_audio, noise_audio, snr_db):
    """
    Adds noise to a clean audio signal at a specific SNR level.

    Args:
        clean_audio (numpy.ndarray): The clean audio waveform.
        noise_audio (numpy.ndarray): The noise audio waveform.
        snr_db (int): The desired Signal-to-Noise Ratio in dB.

    Returns:
        numpy.ndarray: The noisy audio waveform.
    """
    rms_clean = calculate_rms(clean_audio)
    
    if rms_clean == 0:
        return clean_audio

    # Make noise segment the same length as the clean audio
    if len(clean_audio) >= len(noise_audio):
        repeat_factor = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_segment = np.tile(noise_audio, repeat_factor)[:len(clean_audio)]
    else:
        # Take a random snippet of noise if it's longer
        start_index = random.randint(0, len(noise_audio) - len(clean_audio))
        noise_segment = noise_audio[start_index : start_index + len(clean_audio)]

    rms_noise = calculate_rms(noise_segment)
    
    if rms_noise == 0:
        return clean_audio

    # Calculate the required scaling factor for the noise
    snr_linear = 10**(snr_db / 20.0)
    required_rms_noise = rms_clean / snr_linear
    scaling_factor = required_rms_noise / rms_noise

    # Scale the noise and add it to the clean audio
    noisy_audio = clean_audio + (noise_segment * scaling_factor)
    return noisy_audio

if __name__ == "__main__":
    print("--- Starting Noise Augmentation Script ---")

    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    clean_speech_dir = os.path.join(project_root, "data", "clean_german_speech")
    noise_dir = os.path.join(project_root, "data", "noise_samples")
    output_dir = os.path.join(project_root, "data", "augmented_noisy_speech")
    
    # Desired SNR levels to generate
    target_snrs_db = [15, 10, 5, 0, -5]
    
    os.makedirs(clean_speech_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Main Logic ---
    # 1. Get list of clean speech and noise files
    try:
        clean_files = [f for f in os.listdir(clean_speech_dir) if f.endswith('.mp3') or f.endswith('.wav')]
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')or f.endswith('.mp3')]
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e}. Please create the necessary data folders.")
        clean_files, noise_files = [], []


    if not clean_files or not noise_files:
        print("Error: Clean speech or noise sample directories are empty.")
        print(f"Please add clean audio files to: {clean_speech_dir}")
        print(f"Please add noise audio files to: {noise_dir}")
    else:
        print(f"Found {len(clean_files)} clean speech files and {len(noise_files)} noise files.")
        
        # 2. Loop through each clean file and add noise
        for clean_filename in clean_files:
            clean_filepath = os.path.join(clean_speech_dir, clean_filename)
            
            try:
                # Load the clean audio, ensure it's mono and at a consistent sample rate
                clean_audio, sr = librosa.load(clean_filepath, sr=16000, mono=True)
            except Exception as e:
                print(f"Could not load {clean_filename}, skipping. Error: {e}")
                continue

            # Pick a random noise file to mix with
            noise_filename = random.choice(noise_files)
            noise_filepath = os.path.join(noise_dir, noise_filename)
            
            try:
                # Load the noise audio
                noise_audio, _ = librosa.load(noise_filepath, sr=16000, mono=True)
            except Exception as e:
                print(f"Could not load {noise_filename}, skipping. Error: {e}")
                continue

            # 3. Generate a noisy version for each target SNR
            for snr in target_snrs_db:
                print(f"  -> Augmenting '{clean_filename}' with '{noise_filename}' at {snr}dB SNR...")
                
                noisy_audio = add_noise(clean_audio, noise_audio, snr)
                base_name = os.path.splitext(clean_filename)[0]
                output_filename = f"{base_name}_noise_{os.path.splitext(noise_filename)[0]}_snr{snr}.wav"
                output_filepath = os.path.join(output_dir, output_filename)
                
                sf.write(output_filepath, noisy_audio, sr)

        print("\n--- Noise augmentation complete! ---")
        print(f"Noisy files saved in: {output_dir}")