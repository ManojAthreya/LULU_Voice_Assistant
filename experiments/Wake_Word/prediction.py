import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model


fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

def WWD():
    model = load_model("./WWD_Lulu.h5")
    print("Listening for WWD: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    if prediction[:, 1] > 0.75:
        print(f"Wake Word Detected")
        print("Confidence:", prediction[:, 1])
        detection = 1
    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])
        detection = 0
    
    return detection