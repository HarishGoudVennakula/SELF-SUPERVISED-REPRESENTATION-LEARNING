import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load audio file
audio_file = r"audio\output.wav"  # Replace "audio.wav" with the path to your audio file

# Use the recognizer to recognize speech
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)  # Read the entire audio file

# Recognize speech using Google Speech Recognition
try:
    print("Recognizing speech...")
    text = recognizer.recognize_google(audio_data)
    print("Speech recognized: ", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
