def greet(tts_model):
    tts_model.tts_to_file(text="Hi, I'm Lulu, How can I help you today?", file_path="/mnt/c/Ubuntu/virtual_assistant/LuLu/static/TTS_Greet.wav")
    
def speak(tts_model, text_to_speak):
    tts_model.tts_to_file(text=f"{text_to_speak}", file_path="outtext.wav")