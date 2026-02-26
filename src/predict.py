
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

STRESS_WEIGHTS = {
    "neutral": 0.1, "happy": 0.0, "sad": 0.6,
    "angry":   0.9, "fearful": 0.8, "disgust": 0.7
}

def get_stress_level(score):
    if score < 0.3:   return "Low",      "#2ecc71"
    elif score < 0.6: return "Moderate", "#f39c12"
    elif score < 0.8: return "High",     "#e67e22"
    else:             return "Critical", "#e74c3c"

def predict(audio_path, model_path="models/best_model.keras"):
    model    = load_model(model_path)
    features = extract_features(audio_path)
    if features is None:
        return None
    
    inp   = features[np.newaxis, ..., np.newaxis]
    probs = model.predict(inp, verbose=0)[0]
    
    top_idx      = np.argmax(probs)
    emotion      = EMOTION_LABELS[top_idx]
    confidence   = probs[top_idx] * 100
    stress_score = sum(probs[i] * STRESS_WEIGHTS[EMOTION_LABELS[i]]
                      for i in range(len(EMOTION_LABELS)))
    stress_label, color = get_stress_level(stress_score)
    
    return {
        "emotion":      emotion,
        "confidence":   confidence,
        "stress_score": stress_score,
        "stress_level": stress_label,
        "all_probs":    dict(zip(EMOTION_LABELS, probs))
    }

if __name__ == "__main__":
    import sys
    result = predict(sys.argv[1])
    print(result)
