import os
import sys
import pandas as pd
import time

# Setup Project Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import Pipeline Components
from src.audio_utils import load_audio, initialize_vad_model, get_speech_timestamps
from src.transcription import initialize_whisper_model, transcribe_chunk
from src.feature_extractor import (initialize_dictionary, calculate_lexical_validity,
                                   initialize_language_model, calculate_perplexity)

def find_audio_files(directory):
    """Finds all WAV audio files in a given directory and its subdirectories."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def process_single_file_for_features(file_path, vad_model, vad_utils, whisper_model, 
                                     german_dictionary, lm_tokenizer, lm_model):
    """
    Processes a single audio file and extracts features for each speech segment.
    Modified version of the logic in main_pipeline.py.
    """
    all_results = []
    
    waveform, sampling_rate = load_audio(file_path)
    if waveform is None:
        return [] 

    speech_timestamps = get_speech_timestamps(waveform, vad_model, vad_utils, sampling_rate)
    if not speech_timestamps:
        return []

    for i, segment in enumerate(speech_timestamps):
        start_sec = segment['start']
        end_sec = segment['end']
        
        start_sample = int(start_sec * sampling_rate)
        end_sample = int(end_sec * sampling_rate)
        audio_chunk = waveform[start_sample:end_sample]

        if len(audio_chunk) == 0:
            continue

        transcription_result = transcribe_chunk(audio_chunk, whisper_model, language="de")

        if transcription_result and transcription_result['text']:
            transcript_text = transcription_result['text']
            
            features = {
                'avg_logprob': transcription_result['avg_logprob'],
                'lexical_validity': calculate_lexical_validity(transcript_text, german_dictionary),
                'perplexity': calculate_perplexity(transcript_text, lm_tokenizer, lm_model)
            }
            
            final_result = {
                'filepath': file_path,
                'start_time': start_sec,
                'end_time': end_sec,
                'transcription': transcript_text,
                **features
            }
            all_results.append(final_result)
            
    return all_results


if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL_SIZE = "base"
    LM_MODEL_ID = "dbmdz/german-gpt2"
    
    data_root = os.path.join(project_root, "data")
    data_categories = {
        "gcs_2": os.path.join(data_root, "gcs_level_2_incomprehensible"),
        "gcs_3": os.path.join(data_root, "gcs_level_3_word_salad"),
        "gcs_45_clean": os.path.join(data_root, "clean_german_speech"),
        "gcs_45_noisy": os.path.join(data_root, "augmented_noisy_speech"), 
    }
    
    output_csv_path = os.path.join(project_root, "results", "master_feature_dataset.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print("--- Initializing all models for batch processing ---")
    start_init_time = time.time()
    vad_model, vad_utils = initialize_vad_model()
    whisper_model = initialize_whisper_model(model_size=WHISPER_MODEL_SIZE)
    german_dict = initialize_dictionary()
    lm_tokenizer, lm_model = initialize_language_model(model_id=LM_MODEL_ID)
    end_init_time = time.time()
    print(f"--- Models initialized in {end_init_time - start_init_time:.2f} seconds ---\n")

    if not all([vad_model, whisper_model, german_dict, lm_tokenizer, lm_model]):
        print("A model failed to initialize. Aborting batch processing.")
    else:
        # --- Main Batch Processing Logic ---
        all_feature_results = []
        total_start_time = time.time()
        
        for category_name, category_path in data_categories.items():
            print(f"--- Processing category: {category_name} ---")
            print(f"Searching for audio files in: {category_path}")
            
            audio_files_to_process = find_audio_files(category_path)
            
            if not audio_files_to_process:
                print(f"Warning: No .wav files found in '{category_path}'. Skipping.")
                continue
                
            print(f"Found {len(audio_files_to_process)} files to process.")
            
            for i, file_path in enumerate(audio_files_to_process):
                print(f"  -> Processing file {i+1}/{len(audio_files_to_process)}: {os.path.basename(file_path)}")
                
                file_results = process_single_file_for_features(
                    file_path, vad_model, vad_utils, whisper_model,
                    german_dict, lm_tokenizer, lm_model
                )
                
                for result in file_results:
                    result['category'] = category_name
                    if category_name == "gcs_45_noisy" and "snr" in os.path.basename(file_path):
                        try:
                            snr_str = os.path.basename(file_path).split('snr')[1].split('.')[0]
                            result['snr'] = int(snr_str)
                        except (IndexError, ValueError):
                            result['snr'] = None
                    else:
                        result['snr'] = None

                all_feature_results.extend(file_results)

        total_end_time = time.time()
        print(f"\n--- Batch processing complete in {total_end_time - total_start_time:.2f} seconds ---")

        # --- Save to CSV ---
        if all_feature_results:
            print(f"Saving {len(all_feature_results)} total segments to CSV: {output_csv_path}")
            df = pd.DataFrame(all_feature_results)
            
            # Reorder columns for clarity
            column_order = ['filepath', 'category', 'snr', 'start_time', 'end_time', 
                            'transcription', 'avg_logprob', 'lexical_validity', 'perplexity']
            df = df[column_order]
            
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print("Successfully saved master feature dataset.")
        else:
            print("No results were generated. CSV file not saved.")