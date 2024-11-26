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
    QAdata_paths = ['data/WikiQA-train.tsv', 'data/WikiQA-test.tsv', 'data/WikiQA-dev.tsv']
    df = classifier.prepare_data(chat_dataset, QAdata_paths)
    X_test, y_test = classifier.train(df)

#     test_queries = [
#     "What is artificial intelligence?",
#     "How do vaccines work?",
#     "Explain machine learning in simple terms.",
#     "What are the effects of global warming?",
#     "Define quantum computing.",
#     "Who discovered gravity?",
#     "Why do birds migrate?",
#     "How are glacier caves formed?",
#     "What is the capital of France?",
#     "List the top programming languages in 2024.",
#     "What causes earthquakes?",
#     "How does photosynthesis work?",
#     "Who invented the telephone?",
#     "What is the tallest mountain in the world?",
#     "How do computers process information?",
#     "What is the theory of relativity?",
#     "Explain blockchain technology.",
#     "What are black holes?",
#     "Who was Albert Einstein?",
#     "How does the human brain work?",
#     "What is the speed of light?",
#     "What causes tsunamis?",
#     "What is the largest ocean on Earth?",
#     "What is the purpose of the United Nations?",
#     "How do airplanes fly?",
#     "What is climate change?",
#     "How do electric cars work?",
#     "What is the population of the world?",
#     "What is the difference between a star and a planet?",
#     "How does the internet work?",
#     "What is the purpose of democracy?",
#     "What is the Milky Way galaxy?",
#     "Who was Isaac Newton?",
#     "What are the primary colors?",
#     "How do submarines operate?",
#     "What is a solar eclipse?",
#     "How do plants grow?",
#     "What are the basic laws of physics?",
#     "What is gravity?",
#     "What is an atom?",
#     "What are the phases of the moon?",
#     "How do telescopes work?",
#     "What is the history of the Great Wall of China?",
#     "What is the process of photosynthesis?",
#     "What are the uses of DNA?",
#     "What are the different types of ecosystems?",
#     "What is the structure of a cell?",
#     "What is an endangered species?",
#     "What is a black hole?",
#     "What is a galaxy?",
#     "How does electricity work?",
#     "What is the importance of education?",
#     "What are the benefits of renewable energy?",
#     "What are the effects of deforestation?",
#     "What is the water cycle?",
#     "What is the importance of biodiversity?",
#     "How does a computer virus spread?",
#     "What is cybersecurity?",
#     "What are the benefits of artificial intelligence?",
#     "What are the impacts of climate change?",
#     "How does recycling help the environment?",
#     "What is global warming?",
#     "What is the greenhouse effect?",
#     "How do coral reefs form?",
#     "What are the threats to the Amazon rainforest?",
#     "What are the major types of renewable energy?",
#     "What is the function of the human heart?",
#     "What are the main organs in the human body?",
#     "How do satellites orbit the Earth?",
#     "What is the role of gravity in the solar system?",
#     "What are the different layers of the Earth?",
#     "How do volcanoes erupt?",
#     "What is the process of metamorphosis?",
#     "What is the importance of water conservation?",
#     "What is the difference between a lake and a river?",
#     "What is the role of oceans in regulating the climate?",
#     "How do ecosystems maintain balance?",
#     "What are the effects of overfishing?",
#     "What are the differences between reptiles and amphibians?",
#     "How do birds navigate during migration?",
#     "What is the history of the Eiffel Tower?",
#     "What are the most common programming languages?",
#     "What are the benefits of learning Python?",
#     "How does machine learning work?",
#     "What is the role of algorithms in data processing?",
#     "What is the impact of social media on society?",
#     "How does 5G technology work?",
#     "What are the advantages of cloud computing?",
#     "What is the difference between augmented reality and virtual reality?",
#     "What is the purpose of blockchain in cryptocurrencies?",
#     "How does a search engine retrieve information?",
#     "What is the future of artificial intelligence?",
#     "What are the ethical concerns of AI?",
#     "What is the Turing Test?",
#     "What is the impact of automation on jobs?", 
#     "What is the purpose of photosynthesis?",
#     "How do solar panels generate electricity?",
#     "What are the symptoms of climate change?",
#     "Explain the process of water purification.",
#     "What is the history of the Taj Mahal?"
# ]
#     test_chats = [
#     "Hello, how are you?",
#     "What's up?",
#     "Tell me a joke!",
#     "Good morning!",
#     "How's it going?",
#     "Do you like pizza?",
#     "Hi, what's your name?",
#     "Can you recommend a movie?",
#     "Let's talk about something fun.",
#     "What's your favorite color?",
#     "Have you ever been to the beach?",
#     "What's your favorite food?",
#     "Can we be friends?",
#     "Do you know any fun games?",
#     "What's your favorite animal?",
#     "Do you like music?",
#     "What's your favorite song?",
#     "Tell me something interesting.",
#     "Do you like to travel?",
#     "What's your dream destination?",
#     "Have you ever climbed a mountain?",
#     "What's your favorite movie?",
#     "Do you enjoy reading books?",
#     "What's your favorite book?",
#     "Do you have any hobbies?",
#     "What do you like to do for fun?",
#     "What's your favorite season?",
#     "Do you like coffee or tea?",
#     "What did you do today?",
#     "Can you make me laugh?",
#     "Do you enjoy playing sports?",
#     "What's your favorite sport?",
#     "Do you have any pets?",
#     "What kind of pet would you like?",
#     "Do you enjoy spending time outdoors?",
#     "What's your favorite holiday?",
#     "Can you tell me a story?",
#     "What's your favorite ice cream flavor?",
#     "Do you believe in magic?",
#     "What's your favorite thing to do on weekends?",
#     "Do you like rainy days?",
#     "What's your favorite TV show?",
#     "Do you enjoy dancing?",
#     "What's your favorite memory?",
#     "Do you like cooking?",
#     "What's your favorite dish to make?",
#     "Do you play any instruments?",
#     "What's your favorite way to relax?",
#     "Do you enjoy puzzles or riddles?",
#     "What's the funniest thing that happened to you?",
#     "Do you like board games?",
#     "What's your favorite board game?",
#     "Do you enjoy video games?",
#     "What's your favorite video game?",
#     "Do you like art or drawing?",
#     "What's your favorite type of art?",
#     "Have you ever gone camping?",
#     "What's your favorite kind of weather?",
#     "Do you like sunny or cloudy days?",
#     "What makes you happy?",
#     "Can you guess my favorite color?",
#     "Do you enjoy gardening?",
#     "What's your favorite flower?",
#     "Have you ever been on a road trip?",
#     "What's your favorite car?",
#     "Do you enjoy swimming?",
#     "What's your favorite swimming spot?",
#     "Do you enjoy hiking?",
#     "What's your favorite trail?",
#     "Do you like star gazing?",
#     "What's the best view you've ever seen?",
#     "Do you like to watch sunsets?",
#     "What's your favorite time of day?",
#     "Do you like roller coasters?",
#     "What's the most exciting thing you've done?",
#     "Have you ever tried skydiving?",
#     "Do you like comedy shows?",
#     "Who's your favorite comedian?",
#     "Do you enjoy science fiction movies?",
#     "What's your favorite movie genre?",
#     "Do you believe in aliens?",
#     "What's the weirdest thing you've ever seen?",
#     "Have you ever baked a cake?",
#     "What's your favorite dessert?",
#     "Do you enjoy photography?",
#     "What's your favorite photo you've taken?",
#     "Have you ever written a poem?",
#     "Do you like poetry?",
#     "What's your favorite quote?",
#     "Can you guess my favorite hobby?",
#     "What's your favorite drink?",
#     "Do you like milkshakes or smoothies?",
#     "What's your favorite fruit?",
#     "Do you enjoy traveling by train?",
#     "What's your favorite mode of transportation?",
#     "Do you enjoy quiet places?",
#     "What's the quietest place you've ever been?",
#     "Do you like surprises?",
#     "What's the best surprise you've ever received?",
#     "Can you guess what I'm thinking?",
#     "What's your favorite game to play with friends?"
# ]

#     queries = [
#     # Health
#     "What are the most common diseases worldwide?",
#     "What is the leading cause of death globally?",
#     "How can mental health awareness be improved?",
#     "What are the global health statistics for 2024?",
#     "How does stress impact physical health?",
#     "What are the symptoms of diabetes?",
#     "How is cardiovascular disease prevented?",
#     "What are the latest advancements in cancer research?",
#     "What is the role of vaccines in public health?",
#     "How do lifestyle choices affect mental health?",

#     # Environment
#     "What is the primary cause of global warming?",
#     "How does deforestation affect biodiversity?",
#     "What are endangered species and why are they at risk?",
#     "What are the current rates of deforestation globally?",
#     "How does climate change impact agriculture?",
#     "What are the main sources of renewable energy?",
#     "How do coral reefs contribute to the ecosystem?",
#     "What are the effects of air pollution on human health?",
#     "How can we reduce plastic waste in oceans?",
#     "What is the Paris Agreement?",

#     # Technology
#     "What are the latest emerging technologies of 2024?",
#     "How does artificial intelligence impact the job market?",
#     "What is blockchain technology and how does it work?",
#     "How are quantum computers different from classical computers?",
#     "What are the ethical concerns surrounding AI advancements?",
#     "What is the role of 5G in modern communication?",
#     "How is robotics transforming the healthcare industry?",
#     "What are the applications of augmented reality in gaming?",
#     "How does the Internet of Things (IoT) improve daily life?",
#     "What is the future of autonomous vehicles?",

#     # Economy
#     "What factors influence stock market performance?",
#     "How does cryptocurrency impact global economies?",
#     "What is the current state of the global job market?",
#     "How do interest rates affect economic growth?",
#     "What are the top-performing sectors in 2024's economy?",
#     "How is inflation measured in an economy?",
#     "What are the risks of investing in cryptocurrencies?",
#     "How does unemployment affect a country’s GDP?",
#     "What is the role of the World Bank in global development?",
#     "How do trade agreements impact international markets?",

#     # Entertainment
#     "What are the top music streaming platforms in 2024?",
#     "How does social media influence popular culture?",
#     "What are the highest-grossing movies of all time?",
#     "What are the trends in the music industry this year?",
#     "How do awards shows impact entertainment careers?",
#     "What are the most-watched TV series on streaming platforms?",
#     "How has digital media transformed entertainment?",
#     "What is the history of the Grammy Awards?",
#     "How do video games impact mental health?",
#     "What are the effects of reality TV on society?",

#     # Sports
#     "What are the major sporting events of 2024?",
#     "How does data analytics improve sports performance?",
#     "What is the history of the Olympic Games?",
#     "How does sports psychology benefit athletes?",
#     "What are the current rankings in international football?",
#     "How do injuries affect professional athletes?",
#     "What are the top sports leagues in the world?",
#     "How does sponsorship impact sports organizations?",
#     "What is the role of technology in modern sports?",
#     "How does nutrition affect athletic performance?",

#     # Politics
#     "What are the key issues in the upcoming elections?",
#     "How does public policy analysis shape governance?",
#     "What are the impacts of international relations on trade?",
#     "How does voter turnout affect election results?",
#     "What is the role of lobbying in policymaking?",
#     "How do political ideologies influence lawmaking?",
#     "What is the history of the United Nations?",
#     "How do protests influence political decisions?",
#     "What are the effects of sanctions on international relations?",
#     "How does media influence public opinion during elections?",

#     # Education
#     "What are the global literacy rates in 2024?",
#     "How does online education impact traditional learning?",
#     "What are the challenges of student loan debt?",
#     "How does education influence economic development?",
#     "What are the benefits of early childhood education?",
#     "How do education policies affect access to learning?",
#     "What are the trends in e-learning platforms?",
#     "How does technology improve classroom engagement?",
#     "What are the global rankings of universities in 2024?",
#     "How do scholarships help students from low-income families?",

#     # Travel
#     "What are the top tourist destinations in the world?",
#     "How has the airline industry recovered post-pandemic?",
#     "What are the trends in sustainable travel?",
#     "How do travel apps improve trip planning?",
#     "What are the most popular travel destinations in Europe?",
#     "How does tourism impact local economies?",
#     "What are the best destinations for adventure travel?",
#     "How do airlines handle baggage claims?",
#     "What are the most popular budget travel tips?",
#     "What are the benefits of solo travel?",

#     # Food
#     "What are the global crop yield statistics in 2024?",
#     "How does climate change affect food production?",
#     "What are the main causes of global hunger?",
#     "How does food security impact global stability?",
#     "What are the benefits of sustainable farming practices?",
#     "What are the most popular superfoods of 2024?",
#     "How do dietary trends affect food consumption?",
#     "What are the environmental impacts of food waste?",
#     "What are the healthiest cuisines in the world?",
#     "How does agriculture impact water usage?"
# ]
#     chats = [
#     "Hey, how are you?",
#     "What’s up with the global warming stuff?",
#     "Tell me a joke about AI!",
#     "Good morning! Did you hear about the latest tech news?",
#     "Do you like pizza or are you more into healthy food?",
#     "Hi, what’s your favorite movie about sports?",
#     "Can you recommend a good book about mental health?",
#     "Let’s talk about something fun, like traveling to Paris.",
#     "Do you like streaming platforms? What's your favorite show?",
#     "What’s your favorite way to relax after work?",
#     "Do you enjoy hiking? I heard it’s great for mental health.",
#     "How’s the job market looking for data analysts?",
#     "Can we chat about AI and how it’s changing the world?",
#     "Do you believe in aliens or black holes?",
#     "Why do people love traveling so much?",
#     "What's your favorite movie genre?",
#     "What’s a fun thing to do on weekends other than watching sports?",
#     "Did you know the Olympic Games are starting soon?",
#     "Do you prefer road trips or flying?",
#     "Have you tried a plant-based diet? It’s trending a lot now.",
#     "What's your favorite memory of a sporting event?",
#     "How do you spend rainy days? I usually read about history.",
#     "What’s the most interesting tourist spot you’ve visited?",
#     "Do you think cryptocurrencies will replace traditional money?",
#     "Have you ever been on a cruise? I heard they’re amazing.",
#     "Do you like rainy weather? It makes me think about global warming.",
#     "What’s your favorite destination to travel to in winter?",
#     "Have you ever wondered how rockets work?",
#     "What’s your go-to comfort food?",
#     "I’m thinking of learning about blockchain technology. Thoughts?",
#     "Do you like sci-fi movies? They often talk about space exploration.",
#     "What’s your favorite workout for staying healthy?",
#     "Do you enjoy watching debates about elections?",
#     "Have you ever been to a live concert? It’s so thrilling.",
#     "How do you think public opinion shapes political decisions?",
#     "What’s the most exciting tech you’ve seen this year?",
#     "Do you think self-driving cars are safe?",
#     "What’s the weirdest fact you know about the ocean?",
#     "Have you ever visited a coral reef? They’re beautiful.",
#     "What’s your favorite dish to cook for family dinners?",
#     "What’s the funniest meme you’ve seen about AI?",
#     "Do you think sports analytics really help athletes perform better?",
#     "Can we chat about what makes a good teacher?",
#     "What’s your dream destination for a vacation?",
#     "Do you believe in climate change? It’s all over the news.",
#     "What’s your favorite way to celebrate holidays?",
#     "Do you think streaming platforms will replace traditional TV?",
#     "What’s the best app you’ve used for planning travel?",
#     "Can you guess what my favorite sport is?",
#     "What’s your favorite way to spend time outdoors?",
#     "Have you ever been to an endangered species sanctuary?",
#     "What’s your go-to playlist for relaxing?",
#     "Do you like board games? They’re so nostalgic.",
#     "Do you think quantum computing will change everything?",
#     "What’s the best story you’ve read about AI in education?",
#     "Do you enjoy learning about global food security?",
#     "What’s your favorite way to stay updated on technology?",
#     "Do you like stargazing? It’s so calming.",
#     "What’s your favorite thing to watch during the Olympics?",
#     "What’s the weirdest thing you’ve heard about climate change?",
#     "Do you enjoy cooking with exotic ingredients?",
#     "What’s the coolest gadget you’ve used recently?",
#     "Do you think renewable energy will solve our problems?",
#     "Have you ever watched a documentary on deforestation?",
#     "What’s your favorite coffee spot in the city?",
#     "Do you think social media affects mental health?",
#     "What’s your favorite place to watch the sunset?",
#     "Do you enjoy trivia games about history and politics?",
#     "What’s your favorite type of art to look at?",
#     "Do you like long walks in nature? They’re so peaceful.",
#     "Have you ever thought about the importance of water conservation?",
#     "What’s the most interesting recipe you’ve tried?",
#     "What’s your favorite thing about science fiction movies?",
#     "What’s your go-to podcast about current events?",
#     "Have you ever played a sports video game? They’re so fun.",
#     "Do you enjoy reading about elections in other countries?",
#     "What’s the best way to spend a quiet afternoon?",
#     "Do you believe AI will really take over jobs?",
#     "What’s your favorite way to stay active during the day?",
#     "Have you ever thought about the effects of space travel on astronauts?",
#     "What’s your favorite activity when it’s snowing?",
#     "Do you like trying new fitness routines? It’s refreshing.",
#     "What’s your favorite app for learning new skills?",
#     "Do you enjoy solving riddles about economics?",
#     "What’s your favorite memory from a cultural festival?",
#     "What’s your favorite TV show about politics?",
#     "Do you think mental health should be taught in schools?",
#     "What’s the funniest thing you’ve seen about renewable energy?",
#     "Do you think augmented reality will be big in gaming?",
#     "What’s your favorite book about history?",
#     "Do you like traveling to remote destinations?",
#     "What’s your favorite dish from international cuisine?"
# ]
   
    # print("Query Test Results:")
    # classifier.evaluate(queries, 0)  

    # print("Chat Test Results:")
    # classifier.evaluate(chats, 1)