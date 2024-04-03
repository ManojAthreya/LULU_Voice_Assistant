import sounddevice as sd
import librosa
from scipy.io.wavfile import write

filename = "asr.wav"
fs = 16000
seconds = 5

def asr(model, processor):  
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, blocking=True)
    sd.wait()
    write(filename, fs, myrecording)
    audio, sample_rate = librosa.load(filename, sr=16000)
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, language='en')
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

