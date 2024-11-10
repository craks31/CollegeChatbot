from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from chat import chatbot

application = Flask(__name__)


@application.route("/")
def hello():
    return render_template('index.html')


@application.route("/ask", methods=['POST'])
def ask():
    data = request.get_json()
    message = data['messageText']

    # Handle the grammar correction
    corrected_text = TextBlob(message)
    msg = corrected_text.correct()
    #print("CT:", corrected_text.correct())
    bot_response = chatbot(str(msg))

    return jsonify({'status': 'OK', 'answer': bot_response})



if __name__ == "__main__":
    application.run()