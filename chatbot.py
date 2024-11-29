from classifier import PromptClassifier
from chat import ChitChatAPI
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration

class Chatbot:
    def __init__(self, classifier, chit_chat_api, wikipedia_retriever):
        self.classifier = classifier
        self.chit_chat_api = chit_chat_api
        self.wikipedia_retriever = wikipedia_retriever

    def process_input(self, user_input):
        input_type= self.classifier.predict([user_input])[0] # Assuming 0 = "query", 1 = "chit-chat"
        print("input type: ", input_type)
        if input_type == 1:  # Chit-chat
            return self.chit_chat_api.get_response(user_input)
        else:  # Query
            return self.wikipedia_retriever.get_data(user_input)
        
class WikipediaRetriever:
    def get_data(self, query):
        return f"Retrieved Wikipedia information for: {query}"

# Example Usage
if __name__ == "__main__":
    classifier = PromptClassifier()
    chatter = ChitChatAPI()
    wikipedia_retriever = WikipediaRetriever()

    chat_system = Chatbot(classifier, chatter, wikipedia_retriever)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_system.process_input(user_input)
        print(f"Bot: {response}")
