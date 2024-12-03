import json
import os
import pysolr
import requests
import pandas as pd
import requests


class Indexer:
    def __init__(self, core_name: str, vm_ip: str, query_fields: list, field_weights:dict) -> None:
        self.solr_url = f'http://{vm_ip}:8983/solr/'
        self.connection = pysolr.Solr(self.solr_url + core_name, always_commit=True, timeout=5000000)
        self.core_name = core_name
        self.query_fields = query_fields
        self.weights = field_weights

    def delete_core(self): 
        print(os.system('sudo -S su - solr -c "/opt/solr/bin/solr delete -c {core}"'.format(core=self.core_name)))


    def create_core(self): 
        print(os.system(
            'sudo -S su - solr -c "/opt/solr/bin/solr create -c {core} -n data_driven_schema_configs"'.format(
                core=self.core_name)))


    def do_initial_setup(self):
        self.delete_core()
        self.create_core()

    def create_documents(self, docs: dict):
        print(self.connection.add(docs))

    def add_fields(self):
        data = {
            "add-field": [
                {
                    "name": "title",
                    "type": "string",
                    "indexed": True,
                    "multiValued": False
                },

                {
                    "name": "revision_id",
                    "type": "string",
                    "indexed": True,
                    "multiValued": False
                },
                
                {
                    "name": "summary",
                    "type": "text_en",
                    "indexed": True,
                    "multiValued": False,
                    
                },
                {
                    "name": "url",
                    "type": "string",
                    "indexed": True,
                    "multiValued": False
                },
                {
                    "name": "topic",
                    "type": "string",
                    "indexed": True,
                    "multiValued": False
                },
            ]
        }

        print(requests.post(self.solr_url + self.core_name + "/schema", json=data).json())
    
    def query_solr(self, query: str, topics:list, k:int = 10) -> pysolr.Results:

        topic_filter = " OR ".join([f"topic:\"{topic}\"" for topic in topics])
        
        params = {
            'fl': ','.join(self.query_fields + ['score']),
            'rows': k,
            'defType': 'edismax', 
            'qf': ' '.join([f"{field}^{weight}" for field, weight in (self.weights).items()]),
            'fq': topic_filter
        }
        results = self.connection.search(query, **params)
    
        return results

if __name__ == "__main__":
    CORE_NAME = "IRF24P3"
    VM_IP = "localhost"

    query_fields = ["summary", "title"]
    field_weights = {
        "title": 1.0,
        "summary": 3.0
    }
    # Setting up the core and adding the fields
    i = Indexer(CORE_NAME, VM_IP, query_fields, field_weights)
    i.do_initial_setup()
    i.add_fields()

    with open("scraped_data3.json", "r") as f:
        data = json.load(f)

    # Add documents to the index
    i.create_documents(data)

    # Test query
    query = "What is a transformer in machine learning"
    topics = ["Technology"]
    results = i.query_solr(query, topics)
    for result in results:
        title = result.get("title", "[No Title]")
        summary = result.get("summary", "[No Summary]")
        score = result.get("score", 0)
        print(f"- Title: {title}")
        print(f"  Summary: {summary[:200]}...")  # Truncate summary for readability
        print(f"  Relevance Score: {score}\n")

