import os
import random
import soundfile as sf
from gtts import gTTS # gTTS uses Google Translate's TTS service.

def generate_word_salad_audio(output_dir, num_files=20, lang='de'):
    """
    Generates audio files of "word salad" using German words and TTS.

    Args:
        output_dir (str): The directory to save the generated WAV files.
        num_files (int): The number of word salad files to create.
        lang (str): The language for the TTS engine.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Generating {num_files} GCS Level 3 'Word Salad' audio files ---")
    print(f"Saving to: {output_dir}")

    # Need to expand this list for more variety later
    nouns = ["Himmel", "Auto", "Tisch", "Wasser", "Stuhl", "Fenster", "Apfel", "Straße", "Mond", "Sonne"]
    verbs = ["laufen", "singen", "essen", "sehen", "schreiben", "fliegen", "schlafen", "denken", "geben"]
    adjectives = ["schnell", "grün", "laut", "leise", "groß", "klein", "kalt", "warm", "schön"]
    
    all_words = nouns + verbs + adjectives

    for i in range(num_files):
        num_words_in_sentence = random.randint(4, 7)
        salad_words = random.choices(all_words, k=num_words_in_sentence)
        salad_sentence = " ".join(salad_words)
        
        output_filename_mp3 = os.path.join(output_dir, f"gcs3_salad_{i+1:02d}.mp3")
        output_filename_wav = os.path.join(output_dir, f"gcs3_salad_{i+1:02d}.wav")

        try:
            # 1. Use gTTS to create an MP3 file
            tts = gTTS(text=salad_sentence, lang=lang, slow=False)
            tts.save(output_filename_mp3)
            
            # 2. Convert the MP3 to a WAV file (as our pipeline expects WAV)
            # (look into using pydub, maybe better)
            print(f"  Generated '{salad_sentence}' -> Converting to WAV...")
            os.system(f"ffmpeg -i {output_filename_mp3} -ar 16000 -ac 1 {output_filename_wav} -y -hide_banner -loglevel error")
            
            # 3. Clean up the temporary MP3 file
            os.remove(output_filename_mp3)

        except Exception as e:
            print(f"Error generating file for '{salad_sentence}'. Error: {e}")
            print("Please ensure you have an internet connection for gTTS and ffmpeg is installed.")
            # Clean up if mp3 was created but wav failed
            if os.path.exists(output_filename_mp3):
                os.remove(output_filename_mp3)

    print("\n--- Word salad generation complete! ---")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, "data", "gcs_level_3_word_salad")
    generate_word_salad_audio(output_dir=target_dir, num_files=20)