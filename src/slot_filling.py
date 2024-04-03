import json
import argparse
import warnings
# warnings.simplefilter("ignore")

# Define the functions
def get_intent_and_slot(user_query, model, tokenizer, device="cuda"):
    '''Load the Mistral model and get the inference from it'''
    # Define the chat template
    messages = [
        {"role": "user",
         "content": 'Give me only the slots for Intent Detection and slot filling from the text. "%s". The format of your output should only be a dictionary in string format and nothing else with the following characteristics: 1. If the Intent is Music related. The format of the dictionary should be { "Intent": PlayMusic, "SongName": Name of the song, "ArtistName": Name of the artist, } 2. If the Intent is Weather related. The format of the dictionary should be { "Intent": GetWeather, "Location": Name of the song, } 3. If the Intent is Tasks related. The format of the dictionary should be { "Intent": AddTask, "ListTasks": the list of tasks extracted, } 4. If the Intent is News related. The format of the dictionary should be { "Intent": GetNews, "Date": Date in the format YYYY-MM-DD, "Topic": The topic selected, } 5. If the Intent is Time related. The format of the dictionary should be { "Intent": SetTimer, "Time": The time in sec, minutes, or hours, } 6. If the Intent is Call related. The format of the dictionary should be { "Intent": CallPerson, "Name": The name of the person selected, } 6. If the Intent is different from what is defined in the points above. The format of the dictionary should be {}'.lower() % (user_query)}
    ]

    # Apply the chat template and tokenize the input
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    # Push to device
    model_inputs = encodeds.to(device)
    
    # Generate the output
    generated_ids = model.generate(model_inputs, max_new_tokens=120, 
                                   do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    
    # Encode as json
    split_start = decoded[0].find("[/INST]")
    decodable_str = decoded[0][split_start + len("[/INST]"):].replace("\n", "")
    dict_start = decodable_str.find("{")
    dict_end = decodable_str.find("}")
    decodable_str = decodable_str[dict_start: dict_end + 1].replace("\n", "")
    print(decodable_str)
    output = json.loads(decodable_str.strip())
    
    return output