***LULU - A personal Voice Assistant***

This project aims to develop a personal voice assistant leveraging natural language processing (NLP) and artificial intelligence (AI) techniques. The assistant will be capable of performing various tasks such as setting reminders, managing schedules, searching for news, gathering weather information, and more, through voice commands.

***Overview***

The personal voice assistant project involves several key components:

- **Wake Word Detection**: We trained a small Neural Network to detect the wake word 'LULU'. Trained from 170 samples which says LULU from various speakers and 170 background noise and negative samples.

- **Speech Recognition (ASR)**: We implemented speech recognition using the OPEN-AI whisper-tiny model.

- **Intent Detection and Slot Filling**: We harnessed the power of the LLM model Mistral-7B-v0.2 to detect the intents from the ASR and do the slot filling to pass it to the Action fulfilment.

- **Action Fulfilment and Answer Generation (NLG)**: Utilizing web APIs for weather prediction and news data retrieval, LULU seamlessly integrates with external services to provide real-time updates. Additionally, upon integration with compatible devices, LULU executes various actions, delivering tangible results corresponding to user requests.

- **Text-to-Speech (TTS)**: Employing the Coqui TTS model from Hugging Face for text-to-speech synthesis.

***Goals***
- Explore various components of the personal voice assistant architecture.
- Train and implement machine learning models for speech recognition and natural language processing.
- Integrate models into a functional system.
- Gain experience with APIs and database management.
  
***Project Boundaries***
- Limited scope for action fulfilment tasks, focusing on a few straightforward actions.
- Use pre-trained models for ASR and TTS due to data and computational limitations.
- Focus on a few straightforward action fulfilment tasks to reduce complexity.
- Simplify natural language generation using predefined constructs.

***Final Deliverable***

  
The final deliverable will be LULU, a fully functional personal voice assistant designed to enhance user productivity and convenience. 
LULU seamlessly executes tasks such as weather prediction, calling contacts, creating to-do lists, fetching news, and setting timers. 
With its conversational style of interaction, LULU engages users in natural dialogues, providing a personalized and intuitive experience.
Whether it's staying informed, staying organized, or staying connected, LULU revolutionizes the way users interact with their digital assistants, offering unparalleled convenience and efficiency in a compact and user-friendly package.
