import os
import json 
import time
import numpy as np

from audio_utils import load_audio, initialize_vad_model, get_speech_timestamps
from transcription import initialize_whisper_model, transcribe_chunk
from feature_extractor import initialize_dictionary, calculate_lexical_validity, initialize_language_model, calculate_perplexity

def process_audio_file(file_path, vad_model, vad_utils, whisper_model, german_dictionary, lm_tokenizer, lm_model):
    """
    Orchestrates the full pipeline for a single audio file.

    1. Loads audio.
    2. Segments speech using VAD.
    3. Transcribes each segment using Whisper.
    4. Collects and returns the results.

    Args:
        file_path (str): Path to the audio file.
        vad_model: Initialized Silero VAD model.
        vad_utils (dict): Silero VAD utility functions.
        whisper_model: Initialized FasterWhisper model.
        german_dictionary (enchant.Dict): Initialized PyEnchant dictionary for German.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the results for one speech segment.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    print(f"\n--- Starting processing for: {os.path.basename(file_path)} ---")
    
    # 1. Load Audio
    waveform, sampling_rate = load_audio(file_path)
    if waveform is None:
        return []

    # 2. Get Speech Segments
    speech_timestamps = get_speech_timestamps(waveform, vad_model, vad_utils, sampling_rate)
    if not speech_timestamps:
        print("No speech detected in the file.")
        return []
    
    all_results = []
    print(f"Found {len(speech_timestamps)} speech segments. Transcribing now...")

    # 3. Transcribe each segment
    for i, segment in enumerate(speech_timestamps):
        start_sec = segment['start']
        end_sec = segment['end']
        print(f"  -> Processing segment {i+1}/{len(speech_timestamps)} ({start_sec:.2f}s - {end_sec:.2f}s)")
        
        # Convert times to sample indices to slice the audio
        start_sample = int(start_sec * sampling_rate)
        end_sample = int(end_sec * sampling_rate)
        audio_chunk = waveform[start_sample:end_sample]
        
        # Ensure the chunk is not empty
        if len(audio_chunk) == 0:
            print("     Skipping empty audio chunk.")
            continue
        
        # Transcribe the chunk
        transcription_result = transcribe_chunk(audio_chunk, whisper_model, language="de")

        if transcription_result:
            transcript_text = transcription_result['text']
            lexical_score = calculate_lexical_validity(transcription_result['text'], german_dictionary)
            perplexity_score = calculate_perplexity(transcript_text, lm_tokenizer, lm_model)
            final_result = {
                'file': os.path.basename(file_path),
                'segment_index': i + 1,
                'start_time_s': start_sec,
                'end_time_s': end_sec,
                'duration_s': end_sec - start_sec,
                'lexical_validity': lexical_score,
                'perplexity': perplexity_score,
                **transcription_result  # Merges the transcription_result dictionary here
            }
            all_results.append(final_result)
        else:
            print("     Transcription failed for this chunk.")

    print(f"--- Finished processing for: {os.path.basename(file_path)} ---")
    return all_results


if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL_SIZE = "base" # "tiny", "base", "small", "medium"
    LM_MODEL_ID = "dbmdz/german-gpt2"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_audio_path = os.path.join(project_root, "data/CommonVoice21.0/cv-corpus-21.0-2025-03-14/de/clips", "TestSample.mp3")
    output_dir = os.path.join(project_root, "results")
    output_json_path = os.path.join(output_dir, f"results_{WHISPER_MODEL_SIZE}_full_features.json")
    
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Main Execution ---
    pipeline_start_time = time.time()
    
    # 1. Initialize models (do this once)
    print("Initializing models...")
    vad_model, vad_utils = initialize_vad_model()
    whisper_model = initialize_whisper_model(model_size=WHISPER_MODEL_SIZE)
    german_dict = initialize_dictionary()
    lm_tokenizer, lm_model = initialize_language_model(model_id=LM_MODEL_ID)
    print("Models initialized.")

    if vad_model is None or whisper_model is None or german_dict is None or lm_tokenizer is None or lm_model is None:
        print("Failed to initialize one or more models. Exiting.")
    else:
        # 2. Process the audio file
        results = process_audio_file(input_audio_path, vad_model, vad_utils, whisper_model, german_dict, lm_tokenizer, lm_model)

        if results:
            # 3. Save results to a JSON file
            print(f"\nSaving {len(results)} results to {output_json_path}")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print("Results saved successfully.")
            
            # Optional: Print a summary 
            print("\n--- Summary ---")
            for res in results:
                                print(f"Segment {res['segment_index']}: '{res['text']}' "
                      f"(Lex: {res['lexical_validity']:.2f}, "
                      f"PPL: {res['perplexity']:,.0f}, "
                      f"Conf: {res['avg_logprob']:.3f})")
                                

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print(f"\nTotal pipeline execution time: {total_time:.2f} seconds.")