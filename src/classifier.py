import numpy as np

def normalize_feature(value, min_val, max_val, invert=False):
    """
    Normalizes a value to a 0-1 range.
    Clamps the value to be within min_val and max_val.
    """
    clamped_value = np.clip(value, min_val, max_val)

    normalized = (clamped_value - min_val) / (max_val - min_val)
    
    if invert:
        return 1.0 - normalized
    else:
        return normalized

def calculate_combined_comprehensibility_score(features, weights):
    """
    Calculates a combined score from normalized features using given weights.

    Args:
        features (dict): A dictionary with raw feature values, e.g.,
                         {'avg_logprob': -0.5, 'lexical_validity': 0.9, 'perplexity': 150.0}
        weights (dict): A dictionary with weights for each feature, e.g.,
                        {'confidence': 0.4, 'lexical': 0.3, 'perplexity': 0.3}

    Returns:
        float: A combined comprehensibility score between 0.0 and 1.0.
    """
    # --- Define Normalization Ranges (These are initial guesses based on observation!) ---
    # Need to tune these based on actual data later
    # For avg_logprob, less negative is better. A typical range  [-2.0, -0.1]
    CONFIDENCE_MIN = -2.0
    CONFIDENCE_MAX = -0.1
    
    # Perplexity, lower is better. Caped to handle extreme values.
    # A log scale might be better, have to test that later.
    PERPLEXITY_MIN = 10.0 # Anything below this is considered very good
    PERPLEXITY_MAX = 5000.0 # Cap perplexity for normalization
    
    # --- Normalize each feature to a 0-1 scale where 1.0 is "good" ---
    
    # Confidence: Higher is better
    norm_confidence = normalize_feature(
        features.get('avg_logprob', CONFIDENCE_MIN),
        CONFIDENCE_MIN,
        CONFIDENCE_MAX
    )
    
    # Lexical Validity
    norm_lexical = np.clip(features.get('lexical_validity', 0.0), 0.0, 1.0)
    
    # Perplexity: Lower is better
    norm_inv_perplexity = normalize_feature(
        features.get('perplexity', PERPLEXITY_MAX),
        PERPLEXITY_MIN,
        PERPLEXITY_MAX,
        invert=True
    )
    
    # --- Calculate weighted sum ---
    combined_score = (
        norm_confidence * weights.get('confidence', 0.33) +
        norm_lexical * weights.get('lexical', 0.33) +
        norm_inv_perplexity * weights.get('perplexity', 0.34)
    )
    
    return combined_score, {'norm_confidence': norm_confidence, 'norm_lexical': norm_lexical, 'norm_inv_perplexity': norm_inv_perplexity}

def classify_gcs_level(combined_score, no_speech_prob):
    """
    Maps a combined comprehensibility score to a GCS-inspired level.
    These thresholds are initial HEURISTICS and MUST be tuned later.

    Args:
        combined_score (float): The combined score from 0.0 to 1.0.
        no_speech_prob (float): The 'no_speech_prob' from Whisper.

    Returns:
        int: The GCS-inspired classification level (1-5).
    """
    # First, check for a strong "no speech" signal from Whisper or VAD
    # (VAD already segments, but this is a double-check)
    if no_speech_prob > 0.8: # Very high probability of no speech
        return 1

    # Define score thresholds for classification
    # Initial guesses
    THRESHOLD_GCS3 = 0.35 # Score below this might be GCS 2 or 3
    THRESHOLD_GCS4_5 = 0.65 # Score above this is likely coherent speech

    if combined_score >= THRESHOLD_GCS4_5:
        # High score -> High confidence, lexically valid, coherent
        return 5 
    elif combined_score >= THRESHOLD_GCS3:
        # Medium score -> Might be word salad or a flawed sentence
        return 3
    else:
        # Low score -> Likely incomprehensible sounds
        return 2

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing classifier.py ---")

    # Define some feature sets simulating different GCS levels 
    gcs_5_features = {'avg_logprob': -0.2, 'lexical_validity': 1.0, 'perplexity': 50.0, 'no_speech_prob': 0.01}
    gcs_3_features = {'avg_logprob': -0.8, 'lexical_validity': 1.0, 'perplexity': 10000.0, 'no_speech_prob': 0.1}
    gcs_2_features = {'avg_logprob': -2.5, 'lexical_validity': 0.1, 'perplexity': 20000.0, 'no_speech_prob': 0.4}
    
    # Define weights (summing to 1.0)
    # Perplexity has a bit more weight as it's a strong signal for GCS 3
    feature_weights = {'confidence': 0.3, 'lexical': 0.3, 'perplexity': 0.4}

    # Test GCS 5
    score_5, norms_5 = calculate_combined_comprehensibility_score(gcs_5_features, feature_weights)
    level_5 = classify_gcs_level(score_5, gcs_5_features['no_speech_prob'])
    print(f"\nSimulated GCS 5 (Coherent):")
    print(f"  Features: {gcs_5_features}")
    print(f"  Normalized Features: { {k: f'{v:.2f}' for k, v in norms_5.items()} }")
    print(f"  Combined Score: {score_5:.3f} -> Classified as GCS Level: {level_5}")

    # Test GCS 3
    score_3, norms_3 = calculate_combined_comprehensibility_score(gcs_3_features, feature_weights)
    level_3 = classify_gcs_level(score_3, gcs_3_features['no_speech_prob'])
    print(f"\nSimulated GCS 3 (Word Salad):")
    print(f"  Features: {gcs_3_features}")
    print(f"  Normalized Features: { {k: f'{v:.2f}' for k, v in norms_3.items()} }")
    print(f"  Combined Score: {score_3:.3f} -> Classified as GCS Level: {level_3}")

    # Test GCS 2
    score_2, norms_2 = calculate_combined_comprehensibility_score(gcs_2_features, feature_weights)
    level_2 = classify_gcs_level(score_2, gcs_2_features['no_speech_prob'])
    print(f"\nSimulated GCS 2 (Incomprehensible):")
    print(f"  Features: {gcs_2_features}")
    print(f"  Normalized Features: { {k: f'{v:.2f}' for k, v in norms_2.items()} }")
    print(f"  Combined Score: {score_2:.3f} -> Classified as GCS Level: {level_2}")