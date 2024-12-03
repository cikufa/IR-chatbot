from classifier import PromptClassifier
from chat import ChitChatAPI
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration
from flask import Flask, request, jsonify
from flask_cors import CORS
from indexer import Indexer
import json
import logging
import time
from collections import defaultdict


# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class WikipediaRetriever:
    def __init__(self):
        self.CORE_NAME = "IRF24P3"
        self.VM_IP = "localhost"
        self.query_fields = ["summary", "title"]
        self.field_weights = {"title": 1.0, "summary": 3.0}

        logging.info("Initializing Indexer...")
        self.indexer = Indexer(self.CORE_NAME, self.VM_IP, self.query_fields, self.field_weights)

        # Perform initial setup and add fields
        try:
            logging.info("Performing initial setup for Solr indexer...")
            self.indexer.do_initial_setup()
            self.indexer.add_fields()
            logging.info("Fields added successfully.")
        except Exception as e:
            logging.error(f"Error during Solr setup: {e}")

        # Load and index data
        try:
            logging.info("Loading and indexing data from scraped_data.json...")
            with open("scraped_data.json", "r") as f:
                data = json.load(f)
            self.indexer.create_documents(data)  # Add documents to the index
            logging.info("Data indexed successfully.")
        except Exception as e:
            logging.error(f"Error loading or indexing data: {e}")

    def get_data(self, query, topic):
        # logging.debug(f"Querying Solr with query='{query}' and topic='{topic}'...")
        # try:
        #     results = self.indexer.query_solr(query, topic)
        #     if not results:
        #         logging.warning("No results found for the given query.")
        #         return "No relevant results found."

        #     # Format the results
        #     formatted_output = ""
        #     i = 0
        #     for result in results:
        #         i = i +1
        #         title = result.get("title", "[No Title]")
        #         summary = result.get("summary", "[No Summary]")
        #         formatted_output += f"- Title: {title}\n"
        #         formatted_output += f"  Summary: {summary[:200]}...\n"
        #         if i >= 1:
        #             break
        #     return formatted_output
        # except Exception as e:
        #     logging.error(f"Error querying Solr: {e}")
        #     return "An error occurred while retrieving data."

    # Query Solr using the Indexer instance
        results = self.indexer.query_solr(query, topic)

        if not results:
            return "No results found."

        # Find the result with the highest score
        highest_scored_result = max(results, key=lambda x: x.get("score", 0))

        # Extract and format the result
        title = highest_scored_result.get("title", "[No Title]")
        summary = highest_scored_result.get("summary", "[No Summary]")

        formatted_output = f"- Title: {title}\n  Summary: {summary}...\n"  # Truncate summary for readability

        return formatted_output


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
        start_time = time.time()
        try:
            input_type = self.classifier.predict([user_input])[0]  # 0 = Query, 1 = Chit-Chat

            if input_type == 1:  # Chit-chat
                self.metrics["chitchat_count"] += 1
                response = self.chit_chat_api.get_response(user_input)
            else:  # Query
                topic = topics[0] if topics else "General"
                self.metrics["queries_by_topic"][topic] += 1
                self.metrics["total_queries"] += 1
                response = self.wikipedia_retriever.get_data(user_input, topic)

                # Record response time
                elapsed_time = time.time() - start_time
                self.metrics["response_times"][topic].append(elapsed_time)
                self.metrics["min_response_time"] = min(self.metrics["min_response_time"], elapsed_time)
                self.metrics["max_response_time"] = max(self.metrics["max_response_time"], elapsed_time)
                self.metrics["topic_timeline"].append({"timestamp": time.time(), "topic": topic})

            return response
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return "An error occurred while processing your request."

    def get_metrics(self):
        avg_response_times = {
            topic: sum(times) / len(times) if times else 0
            for topic, times in self.metrics["response_times"].items()
        }
        most_popular_topic = max(
            self.metrics["queries_by_topic"], key=self.metrics["queries_by_topic"].get, default="None"
        )

        return {
            "total_queries": self.metrics["total_queries"],
            "chitchat_count": self.metrics["chitchat_count"],
            "queries_by_topic": self.metrics["queries_by_topic"],
            "avg_response_times": avg_response_times,
            "most_popular_topic": most_popular_topic,
            "topic_timeline": self.metrics["topic_timeline"],
            "min_response_time": self.metrics["min_response_time"],
            "max_response_time": self.metrics["max_response_time"],
        }


# Flask app
app = Flask(__name__)
CORS(app)
chat_system = Chatbot()


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        topics = data.get('topics', [])
        user_input = data.get('message', '')

        if user_input.lower() == "exit":
            return jsonify({'response': "Goodbye!"})

        bot_response = chat_system.process_input(user_input, topics)
        return jsonify({'response': bot_response})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An error occurred during processing.'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify(chat_system.get_metrics())
    except Exception as e:
        logging.error(f"Error in metrics endpoint: {e}")
        return jsonify({'error': 'An error occurred while retrieving metrics.'}), 500


if __name__ == '__main__':
    app.run(debug=True)

