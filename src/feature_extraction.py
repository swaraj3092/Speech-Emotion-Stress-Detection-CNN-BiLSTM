
import librosa
import numpy as np

def extract_features(file_path, sr=22050, duration=3,
                     n_mels=128, n_mfcc=40, n_fft=2048,
                     hop_length=512, fixed_length=128):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length)
        
        def fix_length(feature, length=fixed_length):
            if feature.shape[1] < length:
                feature = np.pad(feature, ((0,0),(0, length - feature.shape[1])))
            else:
                feature = feature[:, :length]
            return feature
        
        mel_db   = fix_length(mel_db)
        mfcc     = fix_length(mfcc)
        combined = np.vstack([mel_db, mfcc])
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)
        
        return combined
    except Exception as e:
        print(f"Error: {e}")
        return None
