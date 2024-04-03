import sounddevice as sd
from scipy.io.wavfile import write


def record_audio_and_save(save_path, n_times=25):
    input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i+150) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_background_sound(save_path, n_times=50):
    input("To start recording your background sounds press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2 

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i+100) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

#Step 1: Record yourself saying the Wake Word
# print("Recording the Wake Word:\n")
# record_audio_and_save("audio_data/", n_times=50) 

# Step 2: Record your background sounds (Just let it run, it will automatically record)
# print("Recording the Background sounds:\n")
record_background_sound("background_sound/", n_times=100)