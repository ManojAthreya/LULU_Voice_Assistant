import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import random
import torch
import pprint
import playsound
import subprocess
from src.ui import *
from TTS.api import TTS
from src.slot_filling import *
from src.intent_fullfilling import *
from src.wake_word import *
from src.text_to_speech import *
from src.automatic_speech_detection import *
from sentence_transformers import SentenceTransformer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import warnings
warnings.simplefilter("ignore")

# Define the device globally
device = "cuda"

# Load the model and the tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
intent_slot_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                         quantization_config=bnb_config,
                                                         low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Load the similarity model
full_fillment_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# TTS model
tts_model = TTS('tts_models/en/jenny/jenny', gpu=True, progress_bar=False)

# Load the ASR model
asr_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
asr_model.config.forced_decoder_ids = None

# Load the wakeword model
wake_word_model = load_model("/mnt/c/Ubuntu/virtual_assistant/LuLu/models/WWD_Lulu.h5")


# LuLu responses
lulu_responses = [
    "I am on it!",
    "No worries, I've got your back!",
    "Hmm let me do something on that!",
    "Alright I am on it!",
    "Bruh easy peasy"
]


def wrapper():
    # Remove all the garbage stuff from the terminal
    print("\n" * 100)
    
    # Listen for the wakeword
    wakeword = 0
    greet_enabled = True
    enable_ui = True
    while True:
        if wakeword == 0:
            pprint.pprint("Listening for wakeword......")
            time.sleep(1)
            wakeword = WWD(wake_word_model)
            
            # Greet the user
            if wakeword == 1 and greet_enabled:
                # Enable ui
                if enable_ui:
                    process = subprocess.Popen(["python3", "/mnt/c/Ubuntu/virtual_assistant/LuLu/src/ui.py"])
                    enable_ui = False
                    
                # Gree the user
                greet(tts_model)
                playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/TTS_Greet.wav')
                
                # Disable the greeting
                greet_enabled = False
                
        
        # If wakeword detected run the rest of the pipeline
        try:
            if wakeword==1:        
                # Get the user query
                time.sleep(3)
                print("\n\n")
                time.sleep(1)
                print("Ask you question to LuLu now......")
                user_query = str(asr(asr_model, asr_processor)[0]).strip()
                print(user_query)
                                
                # Get the predicted intent and slotclear
                predicted_intent_and_slot = get_intent_and_slot(user_query=user_query,
                                                                model=intent_slot_model,
                                                                tokenizer=tokenizer,
                                                                device=device)
                print(f"Predicted Intent and the Slot by LuLu : {predicted_intent_and_slot}")
                
                # Edge case 1
                if predicted_intent_and_slot == {}:
                    raise Exception("Edge case 1")
                
                # Response of the VA
                print("\n\n")
                speak(tts_model, random.choice(lulu_responses))
                playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/outtext.wav')
                
                # Action Full-fillment 
                tts_output, task_output = FullFillIntent(json_construct=predicted_intent_and_slot,
                                                        model=full_fillment_model).full_fill_intent()
                
                # Speak the output of the model 
                print("\n\n")
                speak(tts_model, f"{tts_output}")
                playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/outtext.wav')
                
                print("\n\n")
                # Check if worth printing
                if len(task_output) > 0:
                    # Loop and print
                    for i, curr_list in enumerate(task_output, 1):
                        print(f"Result {i}")
                        print(curr_list[0])
                        print(curr_list[1])
                        print(curr_list[2])
                        print("\n")
                
                print("\n\n")
                speak(tts_model, "Is there anything else that I can do for you?")
                playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/outtext.wav')
                
                # Ask the user if they wants to end the converation
                print("\n\n")
                print("Respond in yes or no........")
                user_query = str(asr(asr_model, asr_processor)[0]).strip()
                print(user_query)
                
                # Check if the user does not want to continue
                if "no" in user_query.lower():
                    print("\n\n")
                    speak(tts_model, "Alright see you again!")
                    playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/outtext.wav')
                    
                    # Reset the wakeword and enable the greet
                    wakeword = 0
                    greet_enabled = True
                    
                    # Restart in 5 seconds
                    print("\n\n")
                    print("Restarting the pipeline in 5 seconds....")
                    enable_ui = True
                    process.terminate()
                    time.sleep(5)
                else:
                    time.sleep(5)
                    
            else:
                print("Wake Word Not Detected")
                
        except Exception as e:
            print("\n\n")
            speak(tts_model, f"Sorry I did not understand can you please repeat")
            playsound.playsound('/mnt/c/Ubuntu/virtual_assistant/LuLu/static/outtext.wav')
    
    
if __name__ == '__main__':
    wrapper()