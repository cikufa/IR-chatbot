import wikipedia
from wikipedia.exceptions import DisambiguationError, HTTPTimeoutError, PageError, RedirectError, WikipediaException

import re
import json
from collections import deque
from tqdm import tqdm

import warnings
from bs4 import GuessedAtParserWarning
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore', category=GuessedAtParserWarning)


def scrape_topic(topic: str, start_keywords: list, min_docs: int=500) -> list:
    docs = []
    unique_docs = set()
    seen_pages = set()

    init_pages = []
    for keywords in start_keywords:

        init_pages.append(wikipedia.search(keywords, results=1)[0])
    

    page_queue = deque(init_pages)
    seen_pages.update(init_pages)

    with tqdm(total=min_docs, desc=f"Collecting docs for {topic}", unit="topic", leave=True) as progress_bar:
        while (len(unique_docs) < min_docs) and (page_queue):
            page_name = page_queue.popleft()
            try:
                content = wikipedia.page(page_name, auto_suggest=False)
                docs.append({
                            "title": content.title,
                            "revision_id": content.revision_id,
                            "summary": re.sub(r"[^a-zA-Z0-9\s]", "", content.summary),
                            "url": content.url,
                            "topic": topic
                })

                unique_docs.add(page_name)
                new_pages = set(content.links) - seen_pages
                page_queue.extend(new_pages)
                seen_pages.update(new_pages)
            
                progress_bar.update(1)
            except (DisambiguationError, HTTPTimeoutError, PageError, RedirectError, WikipediaException, KeyError) as e:
                continue

    return docs
        
def parellel_scrape(topics: dict, max_workers: int=  1, min_docs: int = 50) -> list:
    docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for topic, keywords in topics.items():
            futures[executor.submit(scrape_topic, topic, keywords, min_docs)] = topic
           
        for future in as_completed(futures):
                topic_docs = future.result()
                docs.extend(topic_docs)
    return docs



if __name__ == "__main__":

    topics = {
        "Health": ["Common diseases", "lifestyle diseases", "common medicines", "global health statistics"],
        "Environment": ["Global warming", "endangered species", "deforestation rates"],
        "Technology": ["Emerging technologies", "AI advancements",  "green energy", "robotics"],
        "Economy": ["Stock market performance", "job markets", "cryptocurrency trends"],
        "Entertainment": ["Music industry", "popular cultural events", "streaming platforms", "video games"],
        "Sports": ["Major sporting events", "sports analytics", "NFL", "Soccer"],
        "Politics": ["Elections", "public policy analysis", "international relations"],
        "Education": ["Literacy rates", "online education trends", "student loan data", "Professional degrees"],
        "Travel": ["Top tourist destinations", "airline industry data", "travel trends", "united states roadways"],
        "Food": ["Crop yield statistics", "global hunger and food security", "modern foods", "junk foods"]
    }

    # data = []
    min_docs = 5000
    max_workers = 10 
    data = parellel_scrape(topics, max_workers, min_docs)

    with open("scraped_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)