import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pandas as pd


def get_embedding(embedding):
    switch = {
        "openAI": OpenAIEmbeddingFunction()
    }

    return switch.get(embedding)


class RAGCollection(object):
    def __init__(self, embedding_type="openAI", name="RAG"):

        self.name = name
        chroma_client = chromadb.EphemeralClient()
        self.collection = chroma_client.create_collection(name=self.name, embedding_function=get_embedding(embedding_type))

    def add(self, ids, embeddings):

        self.collection.add(ids=ids, documents=embeddings)

    def query_collection(self, query, n_results=10):
        results = self.collection.query(query_texts=query, n_results=n_results, include=['distances'])
        df = pd.DataFrame({
            'id': results['ids'][0],
            'score': results['distances'][0],
        })

        return df
