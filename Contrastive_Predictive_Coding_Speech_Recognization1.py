import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPC = 0

# Function to extract features from a .wav file using the trained CPC model
def extract_features_from_wav(file_path):
    audiodata, _ = librosa.load(file_path, sr=16000)  # Assuming 16kHz sample rate
    audiodata_tensor = torch.tensor(audiodata, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, features, _ = model(audiodata_tensor, model.init_hidden(1, use_gpu=True))

    return features.cpu().numpy()

# Function to recognize speech using Dynamic Time Warping (DTW)
def recognize_speech(audio_features):
    min_distance = np.inf
    recognized_word = None

    for word, template_features in recognized_word.items():
        distance, _ = fastdtw(template_features, audio_features, dist=euclidean)
        if distance < min_distance:
            min_distance = distance
            recognized_word = word

    return recognized_word

# Example usage:
audio_file_path = "test_audio.wav"  # Path to the .wav file to recognize
audio_features = extract_features_from_wav(audio_file_path)
recognized_word = recognize_speech(audio_features)
print("Recognized word:", recognized_word)

# Load the trained CPC model
model = CPC(K=2, seq_len=20480).to(device)
model.load_state_dict(torch.load("weights/CPC_K2.pth"))
model.eval()




