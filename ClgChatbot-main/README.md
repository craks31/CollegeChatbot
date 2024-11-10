# Chatbot
Developed the student chatbot application using NLP, TF-IDF, Cosine similarity and Python Flask.
Refer the requirements.txt file for required packages to install
# data.csv:
Contains two columns query and response
# chatf.py:
-Preprocessed the data
-Computed the maximum similarity using TFIDF and cosine similarity
# chat.py
- Imported class from chatf.py and the method get_response is called
# chatapp.py
- The root route (/): displays the chat interface to the user by rendering the index.html template.
- /ask route, on the other hand, is designed to manage incoming POST requests. It anticipates JSON data containing a messageText field. This route processes the user's input using the chatbot function and sends back the chatbot's response in JSON format.
# webapps.py
- Webscraping code is included rest all is similar to application.py
# webscrap.py
- Similar to chatf.py. Few lines of code are included to execute webscrapping
# index.html
- Chatbot interface



