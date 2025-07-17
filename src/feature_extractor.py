import string
import enchant 

def initialize_dictionary(lang_code="de_DE"):
    """
    Initializes the dictionary for a given language.

    Args:
        lang_code (str): The language code (e.g., "de_DE" for German).

    Returns:
        enchant.Dict: An initialized dictionary object, or None if error.
    """
    try:
        dictionary = enchant.Dict(lang_code)
        print(f"PyEnchant dictionary '{lang_code}' initialized successfully.")
        return dictionary
    except enchant.errors.DictNotFoundError:
        print(f"Error: Dictionary for '{lang_code}' not found.")
        return None

def calculate_lexical_validity(text, german_dictionary):
    """
    Calculates the ratio of valid German words in a given text.

    Args:
        text (str): The transcribed text.
        german_dictionary (enchant.Dict): The initialized PyEnchant dictionary.

    Returns:
        float: The ratio of valid words (0.0 to 1.0).
    """
    if not text or not german_dictionary:
        return 0.0

    # 1. Preprocess the text: remove punctuation and make lowercase
    # Creates a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.lower().translate(translator)

    # 2. Split into words
    words = clean_text.split()
    if not words:
        return 0.0

    # 3. Check each word
    valid_word_count = 0
    for word in words:
        if german_dictionary.check(word):
            valid_word_count += 1
    
    # 4. Calculate ratio
    ratio = valid_word_count / len(words)
    return ratio

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing feature_extractor.py (Lexical Validity) ---")
    
    # Initialize dictionary
    german_dict = initialize_dictionary()

    if german_dict:
        # Test cases
        test_case_1 = "Der schnelle braune Fuchs springt über den faulen Hund."
        test_case_2 = "Das ist ein perfekter Satz"
        test_case_3 = "Himmel viel Auto rennen Straße" # "Word Salad"
        test_case_4 = "asdfgh jkl qwertz" # Gibberish non-words
        test_case_5 = "" # Empty string

        validity_1 = calculate_lexical_validity(test_case_1, german_dict)
        validity_2 = calculate_lexical_validity(test_case_2, german_dict)
        validity_3 = calculate_lexical_validity(test_case_3, german_dict)
        validity_4 = calculate_lexical_validity(test_case_4, german_dict)
        validity_5 = calculate_lexical_validity(test_case_5, german_dict)

        # Not 100% accurate, but should be close to 1.0 for valid sentences
        # Maybe inflected forms are problematic
        # Could improve by using a more advanced NLP library like spaCy or NLTK
        # But for now, this should be sufficient for basic validation
        # Could influence the perfomance otherwise
        print(f"'{test_case_1}' -> Lexical Validity: {validity_1:.2f}")
        print(f"'{test_case_2}' -> Lexical Validity: {validity_2:.2f}")
        print(f"'{test_case_3}' -> Lexical Validity: {validity_3:.2f}") # Should be 1.0
        print(f"'{test_case_4}' -> Lexical Validity: {validity_4:.2f}") # Should be 0.0
        print(f"'{test_case_5}' -> Lexical Validity: {validity_5:.2f}")