import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import chitchat_dataset as chat_data
import numpy as np
# import torch
import joblib
import os

class PromptClassifier:
    def __init__(self, model_path= 'classifier_model.pth'):
        self.vectorizer = CountVectorizer()
        self.classifier = LogisticRegression()
        self.model_path = model_path

        if os.path.isfile(self.model_path):
            self.load_model() 
            print("Model loaded")
        else:
            chat_dataset = chat_data.Dataset()
            QAdata_paths = ['data/WikiQA-train.tsv', 'data/WikiQA-test.tsv', 'data/WikiQA-dev.tsv']
            df = self.prepare_data(chat_dataset, QAdata_paths)
            self.train(df)
            self.save_model()
            print("model trained and saved")

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

    def predict(self, samples):
        print("sample in classifier class", samples)
        X_samples = self.vectorizer.transform(samples)
        return self.classifier.predict(X_samples)

    def evaluate(self, samples, GT):
        predictions = self.predict(samples)
        expected_labels= np.full(len(samples), GT)
        correct_predictions = np.sum(predictions == expected_labels)
        print(f"Correct Predictions: {correct_predictions} out of {len(expected_labels)}")
        return correct_predictions
    
    def save_model(self):
        # torch.save(self.classifier.state_dict(), self.model_path)
        joblib.dump({"classifier": self.classifier, "vectorizer": self.vectorizer}, self.model_path)

    def load_model(self):
        data = joblib.load(self.model_path)
        self.classifier = data["classifier"]
        self.vectorizer = data["vectorizer"]
    
if __name__ == "__main__":
    classifier = PromptClassifier()

    queries = [
    # Health
    "What are the most common diseases worldwide?",
    "What is the leading cause of death globally?",
    "How can mental health awareness be improved?",
    "What are the global health statistics for 2024?",
    "How does stress impact physical health?",
    "What are the symptoms of diabetes?",
    "How is cardiovascular disease prevented?",
    "What are the latest advancements in cancer research?",
    "What is the role of vaccines in public health?",
    "How do lifestyle choices affect mental health?",

    # Environment
    "What is the primary cause of global warming?",
    "How does deforestation affect biodiversity?",
    "What are endangered species and why are they at risk?",
    "What are the current rates of deforestation globally?",
    "How does climate change impact agriculture?",
    "What are the main sources of renewable energy?",
    "How do coral reefs contribute to the ecosystem?",
    "What are the effects of air pollution on human health?",
    "How can we reduce plastic waste in oceans?",
    "What is the Paris Agreement?",

    # Technology
    "What are the latest emerging technologies of 2024?",
    "How does artificial intelligence impact the job market?",
    "What is blockchain technology and how does it work?",
    "How are quantum computers different from classical computers?",
    "What are the ethical concerns surrounding AI advancements?",
    "What is the role of 5G in modern communication?",
    "How is robotics transforming the healthcare industry?",
    "What are the applications of augmented reality in gaming?",
    "How does the Internet of Things (IoT) improve daily life?",
    "What is the future of autonomous vehicles?",

    # Economy
    "What factors influence stock market performance?",
    "How does cryptocurrency impact global economies?",
    "What is the current state of the global job market?",
    "How do interest rates affect economic growth?",
    "What are the top-performing sectors in 2024's economy?",
    "How is inflation measured in an economy?",
    "What are the risks of investing in cryptocurrencies?",
    "How does unemployment affect a country’s GDP?",
    "What is the role of the World Bank in global development?",
    "How do trade agreements impact international markets?",

    # Entertainment
    "What are the top music streaming platforms in 2024?",
    "How does social media influence popular culture?",
    "What are the highest-grossing movies of all time?",
    "What are the trends in the music industry this year?",
    "How do awards shows impact entertainment careers?",
    "What are the most-watched TV series on streaming platforms?",
    "How has digital media transformed entertainment?",
    "What is the history of the Grammy Awards?",
    "How do video games impact mental health?",
    "What are the effects of reality TV on society?",

    # Sports
    "What are the major sporting events of 2024?",
    "How does data analytics improve sports performance?",
    "What is the history of the Olympic Games?",
    "How does sports psychology benefit athletes?",
    "What are the current rankings in international football?",
    "How do injuries affect professional athletes?",
    "What are the top sports leagues in the world?",
    "How does sponsorship impact sports organizations?",
    "What is the role of technology in modern sports?",
    "How does nutrition affect athletic performance?",

    # Politics
    "What are the key issues in the upcoming elections?",
    "How does public policy analysis shape governance?",
    "What are the impacts of international relations on trade?",
    "How does voter turnout affect election results?",
    "What is the role of lobbying in policymaking?",
    "How do political ideologies influence lawmaking?",
    "What is the history of the United Nations?",
    "How do protests influence political decisions?",
    "What are the effects of sanctions on international relations?",
    "How does media influence public opinion during elections?",

    # Education
    "What are the global literacy rates in 2024?",
    "How does online education impact traditional learning?",
    "What are the challenges of student loan debt?",
    "How does education influence economic development?",
    "What are the benefits of early childhood education?",
    "How do education policies affect access to learning?",
    "What are the trends in e-learning platforms?",
    "How does technology improve classroom engagement?",
    "What are the global rankings of universities in 2024?",
    "How do scholarships help students from low-income families?",

    # Travel
    "What are the top tourist destinations in the world?",
    "How has the airline industry recovered post-pandemic?",
    "What are the trends in sustainable travel?",
    "How do travel apps improve trip planning?",
    "What are the most popular travel destinations in Europe?",
    "How does tourism impact local economies?",
    "What are the best destinations for adventure travel?",
    "How do airlines handle baggage claims?",
    "What are the most popular budget travel tips?",
    "What are the benefits of solo travel?",

    # Food
    "What are the global crop yield statistics in 2024?",
    "How does climate change affect food production?",
    "What are the main causes of global hunger?",
    "How does food security impact global stability?",
    "What are the benefits of sustainable farming practices?",
    "What are the most popular superfoods of 2024?",
    "How do dietary trends affect food consumption?",
    "What are the environmental impacts of food waste?",
    "What are the healthiest cuisines in the world?",
    "How does agriculture impact water usage?"
]
    chats = [
    "Hey, how are you?",
    "What’s up with the global warming stuff?",
    "Tell me a joke about AI!",
    "Good morning! Did you hear about the latest tech news?",
    "Do you like pizza or are you more into healthy food?",
    "Hi, what’s your favorite movie about sports?",
    "Can you recommend a good book about mental health?",
    "Let’s talk about something fun, like traveling to Paris.",
    "Do you like streaming platforms? What's your favorite show?",
    "What’s your favorite way to relax after work?",
    "Do you enjoy hiking? I heard it’s great for mental health.",
    "How’s the job market looking for data analysts?",
    "Can we chat about AI and how it’s changing the world?",
    "Do you believe in aliens or black holes?",
    "Why do people love traveling so much?",
    "What's your favorite movie genre?",
    "What’s a fun thing to do on weekends other than watching sports?",
    "Did you know the Olympic Games are starting soon?",
    "Do you prefer road trips or flying?",
    "Have you tried a plant-based diet? It’s trending a lot now.",
    "What's your favorite memory of a sporting event?",
    "How do you spend rainy days? I usually read about history.",
    "What’s the most interesting tourist spot you’ve visited?",
    "Do you think cryptocurrencies will replace traditional money?",
    "Have you ever been on a cruise? I heard they’re amazing.",
    "Do you like rainy weather? It makes me think about global warming.",
    "What’s your favorite destination to travel to in winter?",
    "Have you ever wondered how rockets work?",
    "What’s your go-to comfort food?",
    "I’m thinking of learning about blockchain technology. Thoughts?",
    "Do you like sci-fi movies? They often talk about space exploration.",
    "What’s your favorite workout for staying healthy?",
    "Do you enjoy watching debates about elections?",
    "Have you ever been to a live concert? It’s so thrilling.",
    "How do you think public opinion shapes political decisions?",
    "What’s the most exciting tech you’ve seen this year?",
    "Do you think self-driving cars are safe?",
    "What’s the weirdest fact you know about the ocean?",
    "Have you ever visited a coral reef? They’re beautiful.",
    "What’s your favorite dish to cook for family dinners?",
    "What’s the funniest meme you’ve seen about AI?",
    "Do you think sports analytics really help athletes perform better?",
    "Can we chat about what makes a good teacher?",
    "What’s your dream destination for a vacation?",
    "Do you believe in climate change? It’s all over the news.",
    "What’s your favorite way to celebrate holidays?",
    "Do you think streaming platforms will replace traditional TV?",
    "What’s the best app you’ve used for planning travel?",
    "Can you guess what my favorite sport is?",
    "What’s your favorite way to spend time outdoors?",
    "Have you ever been to an endangered species sanctuary?",
    "What’s your go-to playlist for relaxing?",
    "Do you like board games? They’re so nostalgic.",
    "Do you think quantum computing will change everything?",
    "What’s the best story you’ve read about AI in education?",
    "Do you enjoy learning about global food security?",
    "What’s your favorite way to stay updated on technology?",
    "Do you like stargazing? It’s so calming.",
    "What’s your favorite thing to watch during the Olympics?",
    "What’s the weirdest thing you’ve heard about climate change?",
    "Do you enjoy cooking with exotic ingredients?",
    "What’s the coolest gadget you’ve used recently?",
    "Do you think renewable energy will solve our problems?",
    "Have you ever watched a documentary on deforestation?",
    "What’s your favorite coffee spot in the city?",
    "Do you think social media affects mental health?",
    "What’s your favorite place to watch the sunset?",
    "Do you enjoy trivia games about history and politics?",
    "What’s your favorite type of art to look at?",
    "Do you like long walks in nature? They’re so peaceful.",
    "Have you ever thought about the importance of water conservation?",
    "What’s the most interesting recipe you’ve tried?",
    "What’s your favorite thing about science fiction movies?",
    "What’s your go-to podcast about current events?",
    "Have you ever played a sports video game? They’re so fun.",
    "Do you enjoy reading about elections in other countries?",
    "What’s the best way to spend a quiet afternoon?",
    "Do you believe AI will really take over jobs?",
    "What’s your favorite way to stay active during the day?",
    "Have you ever thought about the effects of space travel on astronauts?",
    "What’s your favorite activity when it’s snowing?",
    "Do you like trying new fitness routines? It’s refreshing.",
    "What’s your favorite app for learning new skills?",
    "Do you enjoy solving riddles about economics?",
    "What’s your favorite memory from a cultural festival?",
    "What’s your favorite TV show about politics?",
    "Do you think mental health should be taught in schools?",
    "What’s the funniest thing you’ve seen about renewable energy?",
    "Do you think augmented reality will be big in gaming?",
    "What’s your favorite book about history?",
    "Do you like traveling to remote destinations?",
    "What’s your favorite dish from international cuisine?"
]
   
    t= ['kk']
    print(classifier)
    print(classifier.predict(t)[0])
    # print("Query Test Results:")
    # classifier.evaluate(queries, 0)  

    # print("Chat Test Results:")
    # classifier.evaluate(chats, 1)