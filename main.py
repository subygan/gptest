from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
import chromadb
#from chromadb.config import Settings
import wget
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import wget
from ast import literal_eval
from rag import RAGCollection

# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

# Note. alternatively you can set a temporary env variable like this:
os.environ["OPENAI_API_KEY"] = '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'
import os
import openai


EMBEDDING_MODEL = "text-embedding-3-small"

# # Ignore unclosed SSL socket warnings - optional in case you get these errors
# import warnings
#
# warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)

# import zipfile
# with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
#     zip_ref.extractall("data")

article_df = pd.read_csv('data/vector_database_wikipedia_articles_embedded.csv')

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

article_df.info(show_counts=True)

chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.
# Uncomment for persistent client
# chroma_client = chromadb.PersistentClient()



if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

wikipedia_content_collection = chroma_client.create_collection(name='wikipedia_content', embedding_function=embedding_function)
wikipedia_title_collection = chroma_client.create_collection(name='wikipedia_titles', embedding_function=embedding_function)


# Add the content vectors
wikipedia_content_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.content_vector.tolist(),
)

# Add the title vectors
wikipedia_title_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.title_vector.tolist(),
)



# collection = RAGCollection(embedding_type="openAI", name="RAG")

# collection.add(ids=article_df.vector_id.tolist(), embeddings=article_df.content_vector.tolist())
#
# results = collection.query_collection(query="modern art in Europe", n_results=10)

results = wikipedia_content_collection.query_collection(query="modern art in Europe", n_results=10)

df = pd.DataFrame({
    'id': results['ids'][0],
    'score': results['distances'][0],
    'title': article_df[article_df.vector_id.isin(results['ids'][0])]['title'],
    'content': article_df[article_df.vector_id.isin(results['ids'][0])]['text'],
})



df