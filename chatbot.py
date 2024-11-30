from classifier import PromptClassifier
from chat import ChitChatAPI
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration
from flask import Flask, request, jsonify
from flask_cors import CORS


class Chatbot:
    def __init__(self):
        self.classifier = PromptClassifier()
        self.chit_chat_api = ChitChatAPI()
        self.wikipedia_retriever =  WikipediaRetriever()

        self.type_cnt = [0,0] #visualization 1. ratio of query to chitchat

    def process_input(self, user_input, topic):
        input_type= self.classifier.predict([user_input])[0] # Assuming 0 = "query", 1 = "chit-chat"
        self.type_cnt[input_type] += 1
        print("input type: ", input_type)
        if input_type == 1:  # Chit-chat
            return self.chit_chat_api.get_response(user_input)
        else:  # Query
            return self.wikipedia_retriever.get_data(user_input, topic)
        
class WikipediaRetriever:
    def get_data(self, query, topic):
        return f"Retrieved Wikipedia information for topic: {topic}"


app = Flask(__name__)
CORS(app)
chat_system = Chatbot()
@app.route('/chat', methods=['POST'])  # This explicitly allows POST requests
def chat():
    data = request.json
    topic = data.get('topic')
    user_input= data.get('message', '') 
    if user_input.lower() == "exit":
        return jsonify({'response':"Goodbye!"})
    bot_response = chat_system.process_input(user_input, topic)
    print("41 happening ")
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
