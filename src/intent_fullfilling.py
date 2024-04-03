import asyncio
import requests
import unicodedata
import python_weather
from tqdm import tqdm
from youtubesearchpython import VideosSearch
from sklearn.metrics.pairwise import cosine_similarity

# Define the script for the PlayMusic
class GeneralClass(object):
    def __init__(self, json_construct, model):
        self.json_construct = json_construct
        self.intents = None
        self.slots = None
        self.model = model

    def _correct_intent(self, dict_type):
        # Collect the old intent
        old_intent = self.parse_dict["Intent"]
        return self._get_best_match(self.model.encode(old_intent).reshape(1, -1),
                                    self.model.encode(self.intents), dict_type)

    def _get_best_match(self, query_embedding, passage_embedding, dict_type="slots"):
        # Encode the similarity
        _similarity = list(cosine_similarity(query_embedding,
                                             passage_embedding).reshape(-1))

        # Get the max similarity and its index
        if dict_type == "slots":
            return self.slots[_similarity.index(max(_similarity))]
        else:
            return self.intents[_similarity.index(max(_similarity))]
    
    def _parse_json(self):
        # Perform the key matching
        matched_keys = {}

        # Define all the comparison keys
        passage_embedding = self.model.encode(self.slots)

        # Loop
        for key in self.json_construct.keys():
            # Encode the query
            query_embedding = self.model.encode(key)

            # Encode the similarity
            best_match = self._get_best_match(query_embedding.reshape(1, -1),
                                              passage_embedding, dict_type="slots")
            
            # Update the data
            matched_keys[key] = best_match

        # Make the new dict
        self.parse_dict = {}
        for key, val in matched_keys.items():
            self.parse_dict[val] = self.json_construct[key]

        # Fix the Intent name
        self.parse_dict["Intent"] = self._correct_intent("intents")
        
        
# Define the script for the PlayMusic
class FullFillIntent(GeneralClass):
    def __init__(self, json_construct, model):
        # Init the super class
        super(FullFillIntent, self).__init__(None, model)

        # Init the data
        self.json_construct = json_construct
        self.intents = ["PlayMusic", "GetWeather", "AddTask", "GetNews", "CallPerson", "SetTimer"]
        self.slots = ["Intent", "SongName", "ArtistName", "Location", "TaskList", "Date", "Topic", "Name", "Time"]

    def full_fill_intent(self):
        # Parse the json
        self._parse_json()
        print(self.parse_dict)
        dict_slots_intent = self.parse_dict
        
        # Choose the appropiate choice
        if dict_slots_intent["Intent"] == "PlayMusic":
            return self._play_music_results()
        elif dict_slots_intent["Intent"] == "GetWeather":
            return asyncio.run(self._get_weather_results())
        elif dict_slots_intent["Intent"] == "AddTask":
            return self._add_task_results()
        elif dict_slots_intent["Intent"] == "GetNews":
            return self._fetch_news_results()
        elif dict_slots_intent["Intent"] == "SetTimer":
            return self._set_timer()
        elif dict_slots_intent["Intent"] == "CallPerson":
            return self._call_person()
        else:
            print("No suitable action found.....")
            
    def _call_person(self):
        return f"Calling {self.parse_dict['Name']}", []
    
    def _set_timer(self):
        time = self.parse_dict["Time"]
        return f"Okay setting a timer for {time}", []

    def _play_music_results(self, k=2):
        # Fetch the results
        search_result = VideosSearch(f'{self.parse_dict["SongName"]} by {self.parse_dict["ArtistName"]}',
                                     limit=10)

        # Loop and collect results
        list_important_data = []
        for vid in search_result.result()["result"][:k]:
            current_list = ["https://www.youtube.com/watch?v=" + vid["id"], vid["title"], vid["thumbnails"][-1]["url"]]
            list_important_data.append(current_list)

        return "Here are the result that I found on youtube", list_important_data

    async def _get_weather_results(self):
        # Define the client
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            # fetch a weather forecast from a city
            weather = await client.get(self.parse_dict["Location"])
    
        return f"The current weather in {self.parse_dict['Location']} is {weather.description.lower()} with a current temperature of {weather.temperature} but it feels like {weather.feels_like}", []
    
    def _add_task_results(self):
        # Placeholder
        # task_list = []
        
        # Write the task to the list
        with open("/mnt/c/Ubuntu/virtual_assistant/Pipelines/results.txt", "a+") as f:
            # Write the content to the file
            for line in self.parse_dict['TaskList']:
                f.write(line + '\n')
            
            # Go to the start of the file
            f.seek(0)
    
            # Read and print the file contents
            print("\n\n")
            for line in f:
                print(line)
                # task_list.append(line.strip())
            
        return "Added the task to your list of tasks", []
    
    def _fetch_news_results(self, k=2):
        # Placeholder
        news_data = []
        
        # Fetch the variables
        date_range = self.parse_dict["Date"].lower().strip()
        search_string = "-".join(self.parse_dict["Topic"].split()).lower().strip()
        
        # Fetch the response
        response = requests.get(f"https://newsapi.org/v2/everything?q={search_string}&from={date_range}&sortBy=publishedAt&apiKey=40d7946cb20f4078bfb9c88f15edd8f5")
        if response.status_code == 200:
            for news in response.json()["articles"][:k]:
                title_curr = " ".join(unicodedata.normalize("NFKD", news["title"]).split())
                description = " ".join(unicodedata.normalize("NFKD", news["description"]).split())
                url = news["url"]
                news_data.append([title_curr, description, url])
                
        return f"Here is the news for {search_string} after {date_range} that I found on the web.", news_data