# Erika - Conversational AI

## About:
   Erika is a versatile chatbot trained to answer all the questions related to multiple domains such as general science, data science, history, geography, news, and sports. (More can be added.)

## Design:
  - Created an intents.json file which will help classify the user prompts, it is structured in the format:
    - Tags: It represents the user prompt class (ex. 'Hi, how are you' -- tag: greet). 
    - Patterns: It contains all the potential questions the user can ask the bot.
    - Response: It contains all the possible responses to the user's prompt.
  - Loaded the intents file and preprocessed all the words using NLP (Natural Language Processing) through steps such as tokenization, stemming, and lemmatization, and store all the words and classes of responses.
  - Created training data by performing the following steps
    - lemmatize all the selected words.
    - initialize a bag of words including all the unique words.
    - select all the unique classes
  - Then, we created our sequential model using stacks of two layers **Dense** and **Dropout**, and used **SGD (Stochastic Gradient Descent)** as an optimizer.
  - Initialized the input shape of the first dense node to be the shape of our training data.
  - Successively increased the node size of each dense layer by 2 times (i.e 64, 128) and set each activation function as ReLu.
  - For the last layer, i.e the output layer, the no. of nodes will be equal to the no. of predicted classes, thus we set the activation function to be **softmax**
  - Saved the model into a .h5 file later to be retrieved for the predictions  

## Installation:

  1. Go to the project directory:
     ```
     cd ChatBot
     ```
  2. Install all the dependencies:
     ```
     pip install -r requirements.txt
     ```
  3. Execute the streamlit app:
     ```
     streamlit run run_bot.py
     ```
  
