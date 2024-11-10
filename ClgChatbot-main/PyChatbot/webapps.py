from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from webscrap import QASys
import requests
from bs4 import BeautifulSoup
import pandas as pd
# Example URL
url = "https://www.umkc.edu/admissions/apply/supporting-documents.html"
responses_list = []
# Send a GET request to the URL
response = requests.get(url)
# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Extract relevant information
# Example: Extract text from all paragraph tags
paragraphs = soup.find_all('p')
text_data = df = pd.DataFrame([p.get_text() for p in paragraphs])
contents_div = soup.find(id="contents")

# Extract and print text from the found elements
if contents_div:
    headers = contents_div.find_all(['h3', 'h4'])
     # Iterate through headers
    for header in headers:
        # Extract text from the header
        query_text = header.get_text(strip=True)
        # Find the next elements after the header (either <p>, <ul>, or <ol>)
        next_elements = header.find_all_next(['p', 'ul', 'ol'])
        # Iterate through next elements and store each response in the list
        for element in next_elements:
            if element.name == 'ul' or element.name == 'ol':
                # Iterate through list items in the element
              for list_item in element.find_all('li'):
                    response_text = list_item.get_text(strip=True)

                    # Check if the response has more than 3 words
                    if len(response_text.split()) > 3 and '|' not in response_text :
                        # Append the query_text and response_text to the list
                        response_text = response_text.replace('\t', '').replace('\n', '')
                        ' '.join((response_text).split())
                        responses_list.append((query_text+response_text, query_text+" : "+response_text))
            else:
                response_text = element.get_text(strip=True)

                # Check if the response has more than 4 words
                if len(response_text.split()) > 4 and '|' not in response_text :
                    # Append the query_text and response_text to the list
                    response_text = response_text.replace('\t', '').replace('\n', '')
                    ' '.join((response_text).split())
                    responses_list.append((query_text+" "+response_text, query_text+" : "+response_text))

        # Print the final list of responses
        for response in responses_list:
          print(response)
train = QASys('data.csv', responses_list)
application = Flask(__name__)
@application.route("/")
def hello():
    return render_template('index.html')

@application.route("/ask", methods=['POST'])
def ask():
    data = request.get_json()
    message = data['messageText']

    corrected_text = TextBlob(message)
    msg = corrected_text.correct()
    #print("CT:", corrected_text.correct())
    bot_response =train.get_response(str(msg))

    return jsonify({'status': 'OK', 'answer': bot_response})

if __name__ == "__main__":
    application.run(port=80)