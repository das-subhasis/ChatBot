# Importing the libraries we need
import numpy
import nltk
import numpy as np
import streamlit as st
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random
import os

lemmatizer = WordNetLemmatizer()

# loading all the packages
model = load_model('erika_model1.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json', 'r').read())


# clean the sentence received from the user
def format_sentence(sentence: str) -> list[str]:
    tokens = nltk.tokenize.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w) for w in tokens if w in words]
    return sentence_words


# return bag of words [ 0 , 1 ] for each word in the bag that exists in the sentence
def bow(sentence: str, words: list[str], verbose: bool = False):
    sentence_words = format_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for idx, word in enumerate(words):
            if w == word:
                bag[idx] = 1
                if verbose:
                    print(f'Selected word for bow : \n {word}')

    return np.array(bag)


# predict the intent of the response
def predict_class(sentence: str, verbose: bool = False) -> list[dict[str | None, int]]:
    feature_words = bow(sentence, words)
    pred = model.predict(np.array([feature_words]))[0]
    ERROR_THRESH = 0.25
    # keep the response which is above the threshold
    results = [[i, res] for i, res in enumerate(pred) if res > ERROR_THRESH]
    # sort the result by the strength of probability
    results.sort(key=lambda res: res[1], reverse=True)
    return_list = []
    for res in results:
        return_list.append({"intent": classes[res[0]], "probability": res[1]})
    if verbose:
        print(f"Prediction Results: \n")
        for intent in return_list:
            print(intent)
    # check whether the predictions produced results or not
    if return_list:
        return return_list
    return [{"intent": None, "probability": 0}]


# fetch the response from the intents.json file
def get_response(intents_list, intents_json):
    predicted_tag = intents_list[0]['intent']
    if not predicted_tag:
        error_response = ["sorry, could you say that again?.", "sorry, i am unable to understand that."]
        return random.choice(error_response)
    for intent in intents_json['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response


# fetch the response based on the user input
def chat_response(msg):
    intents_list = predict_class(msg)
    response = get_response(intents_list, intents)
    return response


def main():
    st.title('Erika.AI')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("Talk with Erika"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = chat_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
