import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import chitchat_dataset as chat_data
import numpy as np

class PromptClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = LogisticRegression()

    def clean_data(self, df, text_column="text", label_column="label"):
        df = df[1:]
        df = df.drop_duplicates(subset=[text_column, label_column]).reset_index(drop=True)
        return df

    def prepare_data(self, chat_dataset, QAdata_paths):
        # Prepare chat data
        chat_data = []
        for convo_id, convo in chat_dataset.items():
            chat_data.append({"text": convo['messages'][0][0]['text'], 'label': "chit-chat"})
        chat_df = self.clean_data(pd.DataFrame(chat_data))
        chat_df = chat_df.iloc[1200:].reset_index(drop=True)

        # Prepare query data
        query_data = []
        for path in QAdata_paths:
            try:
                with open(path, 'r') as file:
                    for line in file:
                        query_data.append({"text": line.split("\t")[1], 'label': "Query"})
            except Exception as e:
                print(f"An error occurred while reading {path}: {e}")
        query_df = self.clean_data(pd.DataFrame(query_data))

        # Combine and shuffle datasets
        combined_df = pd.concat([query_df, chat_df], ignore_index=True)
        combined_df = shuffle(combined_df).reset_index(drop=True)

        # Map labels
        combined_df['label'] = combined_df['label'].map({'Query': 0, 'chit-chat': 1})

        return combined_df

    def train(self, df):
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Training Accuracy: {accuracy:.4f}")
        return X_test, y_test

    def predict(self, samples):
        X_samples = self.vectorizer.transform(samples)
        return self.classifier.predict(X_samples)

    def evaluate(self, samples, GT):
        predictions = self.predict(samples)
        expected_labels= np.full(len(samples), GT)
        correct_predictions = np.sum(predictions == expected_labels)
        print(f"Correct Predictions: {correct_predictions} out of {len(expected_labels)}")
        return correct_predictions

if __name__ == "__main__":
    classifier = PromptClassifier()
    chat_dataset = chat_data.Dataset()
    QAdata_paths = ['WikiQACorpus/WikiQA-train.tsv', 'WikiQACorpus/WikiQA-test.tsv', 'WikiQACorpus/WikiQA-dev.tsv']
    df = classifier.prepare_data(chat_dataset, QAdata_paths)
    X_test, y_test = classifier.train(df)

    test_queries = [
    "What is artificial intelligence?",
    "How do vaccines work?",
    "Explain machine learning in simple terms.",
    "What are the effects of global warming?",
    "Define quantum computing.",
    "Who discovered gravity?",
    "Why do birds migrate?",
    "How are glacier caves formed?",
    "What is the capital of France?",
    "List the top programming languages in 2024.",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "Who invented the telephone?",
    "What is the tallest mountain in the world?",
    "How do computers process information?",
    "What is the theory of relativity?",
    "Explain blockchain technology.",
    "What are black holes?",
    "Who was Albert Einstein?",
    "How does the human brain work?",
    "What is the speed of light?",
    "What causes tsunamis?",
    "What is the largest ocean on Earth?",
    "What is the purpose of the United Nations?",
    "How do airplanes fly?",
    "What is climate change?",
    "How do electric cars work?",
    "What is the population of the world?",
    "What is the difference between a star and a planet?",
    "How does the internet work?",
    "What is the purpose of democracy?",
    "What is the Milky Way galaxy?",
    "Who was Isaac Newton?",
    "What are the primary colors?",
    "How do submarines operate?",
    "What is a solar eclipse?",
    "How do plants grow?",
    "What are the basic laws of physics?",
    "What is gravity?",
    "What is an atom?",
    "What are the phases of the moon?",
    "How do telescopes work?",
    "What is the history of the Great Wall of China?",
    "What is the process of photosynthesis?",
    "What are the uses of DNA?",
    "What are the different types of ecosystems?",
    "What is the structure of a cell?",
    "What is an endangered species?",
    "What is a black hole?",
    "What is a galaxy?",
    "How does electricity work?",
    "What is the importance of education?",
    "What are the benefits of renewable energy?",
    "What are the effects of deforestation?",
    "What is the water cycle?",
    "What is the importance of biodiversity?",
    "How does a computer virus spread?",
    "What is cybersecurity?",
    "What are the benefits of artificial intelligence?",
    "What are the impacts of climate change?",
    "How does recycling help the environment?",
    "What is global warming?",
    "What is the greenhouse effect?",
    "How do coral reefs form?",
    "What are the threats to the Amazon rainforest?",
    "What are the major types of renewable energy?",
    "What is the function of the human heart?",
    "What are the main organs in the human body?",
    "How do satellites orbit the Earth?",
    "What is the role of gravity in the solar system?",
    "What are the different layers of the Earth?",
    "How do volcanoes erupt?",
    "What is the process of metamorphosis?",
    "What is the importance of water conservation?",
    "What is the difference between a lake and a river?",
    "What is the role of oceans in regulating the climate?",
    "How do ecosystems maintain balance?",
    "What are the effects of overfishing?",
    "What are the differences between reptiles and amphibians?",
    "How do birds navigate during migration?",
    "What is the history of the Eiffel Tower?",
    "What are the most common programming languages?",
    "What are the benefits of learning Python?",
    "How does machine learning work?",
    "What is the role of algorithms in data processing?",
    "What is the impact of social media on society?",
    "How does 5G technology work?",
    "What are the advantages of cloud computing?",
    "What is the difference between augmented reality and virtual reality?",
    "What is the purpose of blockchain in cryptocurrencies?",
    "How does a search engine retrieve information?",
    "What is the future of artificial intelligence?",
    "What are the ethical concerns of AI?",
    "What is the Turing Test?",
    "What is the impact of automation on jobs?", 
    "What is the purpose of photosynthesis?",
    "How do solar panels generate electricity?",
    "What are the symptoms of climate change?",
    "Explain the process of water purification.",
    "What is the history of the Taj Mahal?"
]
    test_chats = [
    "Hello, how are you?",
    "What's up?",
    "Tell me a joke!",
    "Good morning!",
    "How's it going?",
    "Do you like pizza?",
    "Hi, what's your name?",
    "Can you recommend a movie?",
    "Let's talk about something fun.",
    "What's your favorite color?",
    "Have you ever been to the beach?",
    "What's your favorite food?",
    "Can we be friends?",
    "Do you know any fun games?",
    "What's your favorite animal?",
    "Do you like music?",
    "What's your favorite song?",
    "Tell me something interesting.",
    "Do you like to travel?",
    "What's your dream destination?",
    "Have you ever climbed a mountain?",
    "What's your favorite movie?",
    "Do you enjoy reading books?",
    "What's your favorite book?",
    "Do you have any hobbies?",
    "What do you like to do for fun?",
    "What's your favorite season?",
    "Do you like coffee or tea?",
    "What did you do today?",
    "Can you make me laugh?",
    "Do you enjoy playing sports?",
    "What's your favorite sport?",
    "Do you have any pets?",
    "What kind of pet would you like?",
    "Do you enjoy spending time outdoors?",
    "What's your favorite holiday?",
    "Can you tell me a story?",
    "What's your favorite ice cream flavor?",
    "Do you believe in magic?",
    "What's your favorite thing to do on weekends?",
    "Do you like rainy days?",
    "What's your favorite TV show?",
    "Do you enjoy dancing?",
    "What's your favorite memory?",
    "Do you like cooking?",
    "What's your favorite dish to make?",
    "Do you play any instruments?",
    "What's your favorite way to relax?",
    "Do you enjoy puzzles or riddles?",
    "What's the funniest thing that happened to you?",
    "Do you like board games?",
    "What's your favorite board game?",
    "Do you enjoy video games?",
    "What's your favorite video game?",
    "Do you like art or drawing?",
    "What's your favorite type of art?",
    "Have you ever gone camping?",
    "What's your favorite kind of weather?",
    "Do you like sunny or cloudy days?",
    "What makes you happy?",
    "Can you guess my favorite color?",
    "Do you enjoy gardening?",
    "What's your favorite flower?",
    "Have you ever been on a road trip?",
    "What's your favorite car?",
    "Do you enjoy swimming?",
    "What's your favorite swimming spot?",
    "Do you enjoy hiking?",
    "What's your favorite trail?",
    "Do you like star gazing?",
    "What's the best view you've ever seen?",
    "Do you like to watch sunsets?",
    "What's your favorite time of day?",
    "Do you like roller coasters?",
    "What's the most exciting thing you've done?",
    "Have you ever tried skydiving?",
    "Do you like comedy shows?",
    "Who's your favorite comedian?",
    "Do you enjoy science fiction movies?",
    "What's your favorite movie genre?",
    "Do you believe in aliens?",
    "What's the weirdest thing you've ever seen?",
    "Have you ever baked a cake?",
    "What's your favorite dessert?",
    "Do you enjoy photography?",
    "What's your favorite photo you've taken?",
    "Have you ever written a poem?",
    "Do you like poetry?",
    "What's your favorite quote?",
    "Can you guess my favorite hobby?",
    "What's your favorite drink?",
    "Do you like milkshakes or smoothies?",
    "What's your favorite fruit?",
    "Do you enjoy traveling by train?",
    "What's your favorite mode of transportation?",
    "Do you enjoy quiet places?",
    "What's the quietest place you've ever been?",
    "Do you like surprises?",
    "What's the best surprise you've ever received?",
    "Can you guess what I'm thinking?",
    "What's your favorite game to play with friends?"
]

    print("Query Test Results:")
    classifier.evaluate(test_queries, 0)  

    print("Chat Test Results:")
    classifier.evaluate(test_chats, 1) 
