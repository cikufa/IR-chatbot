from classifier import PromptClassifier
from chat import ChitChatAPI
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration
from flask import Flask, request, jsonify
from flask_cors import CORS


import time
from collections import defaultdict


class Chatbot:
    def __init__(self):
        self.classifier = PromptClassifier()
        self.chit_chat_api = ChitChatAPI()
        self.wikipedia_retriever = WikipediaRetriever()

        # Visualization Metrics
        self.metrics = {
            "total_queries": 0,
            "chitchat_count": 0,
            "queries_by_topic": defaultdict(int),
            "response_times": defaultdict(list),  
            "topic_timeline": [],  
            "min_response_time": float('inf'), 
            "max_response_time": float('-inf')  
        }

    def process_input(self, user_input, topics):
        start_time = time.time()  # Start timer for response time
        input_type = self.classifier.predict([user_input])[0]  # 0 = Query, 1 = Chit-Chat

        if input_type == 1:  # Chit-chat
            self.metrics["chitchat_count"] += 1
            response = self.chit_chat_api.get_response(user_input)
        
        else:  # Query
            # topic = topics[0] if topics else "General"  # Use the first topic or "General"
            elapsed_time = time.time() - start_time
            for topic in topics:
                self.metrics["queries_by_topic"][topic] += 1
                self.metrics["topic_timeline"].append({
                "timestamp": time.time(),
                "topic": topic
            })
                self.metrics["response_times"][topic].append(elapsed_time)
            
            self.metrics["total_queries"] += 1
            self.metrics["min_response_time"] = min(self.metrics["min_response_time"], elapsed_time)
            self.metrics["max_response_time"] = max(self.metrics["max_response_time"], elapsed_time)
            response = self.wikipedia_retriever.get_data(user_input, topics)

        return response

    def get_metrics(self):
        # Calculate average response time per topic
        avg_response_times = {
            topic: sum(times) / len(times) if times else 0
            for topic, times in self.metrics["response_times"].items()
        }

        # Determine the most popular topic
        most_popular_topic = max(
            self.metrics["queries_by_topic"], key=self.metrics["queries_by_topic"].get, default="None"
        )

        return {
            "total_queries": self.metrics["total_queries"],
            "chitchat_count": self.metrics["chitchat_count"],
            "queries_by_topic": self.metrics["queries_by_topic"],
            "avg_response_times": avg_response_times,
            "most_popular_topic": most_popular_topic,
            "topic_timeline": self.metrics["topic_timeline"],  # Return the timeline
            "min_response_time": self.metrics["min_response_time"],  # Min response time
            "max_response_time": self.metrics["max_response_time"],  # Max response time
        }

       
class WikipediaRetriever:
    def get_data(self, query, topics):
        return f"Retrieved Wikipedia information for topic: {topics}"


app = Flask(__name__)
CORS(app)
chat_system = Chatbot()
@app.route('/chat', methods=['POST'])  # This explicitly allows POST requests
def chat():
    data = request.json
    topics = data.get('topics', [])     
    user_input= data.get('message', '') 
    if user_input.lower() == "exit":
        return jsonify({'response':"Goodbye!"})
    bot_response = chat_system.process_input(user_input, topics)
    return jsonify({'response': bot_response})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(chat_system.get_metrics())

if __name__ == '__main__':
    app.run(debug=True)




