from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from datasets import load_dataset
import pandas as pd
from llama_index.core.schema import MetadataMode
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import tqdm

load_dotenv()

# ----- DATA CLEANING ----- #

# Download giftcards products and reviews
giftcards_reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Gift_Cards", trust_remote_code=True)
giftcards_meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Gift_Cards", trust_remote_code=True)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Convert to pandas df
giftcards_reviews_df = pd.DataFrame(giftcards_reviews_dataset['full'][0:500])
giftcards_meta_df = pd.DataFrame(giftcards_meta_dataset['full'][0:500])

# For each product, add all relevant reviews.
# Combine all reviews for a given parent ASIN.
giftcards_reviews_df = giftcards_reviews_df.groupby('parent_asin').agg({'text': '\n'.join}).reset_index()
giftcards_reviews_df = giftcards_reviews_df.merge(giftcards_meta_df, on='parent_asin')

# Create a new df with only the relevant columns
giftcards_df = giftcards_reviews_df[['parent_asin', 'text', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'details', 'subtitle']]

# Convert columns to types that the LLM can handle
giftcards_df.loc[:, "images"] = giftcards_df["images"].apply(lambda x: x.get("hi_res", None) if isinstance(x, dict) else None)
giftcards_df.loc[:, "images"] = giftcards_df["images"].apply(lambda x: x[0] if isinstance(x, list) else x)
giftcards_df.loc[:, "features"] = giftcards_df["features"].apply(lambda x: '\n'.join(x) if isinstance(x, list) else x)

# ----- LLM PREP ----- #

# Convert df to list of dicts
giftcards_data = giftcards_df.to_dict(orient='records')

# Pop unsupported columns from the dict.
for product in giftcards_data:
    product.pop("description")

# Create LlamaIndex documents for each review
documents = []
for review in giftcards_data:
    documents.append(
        Document(
            text=review["title"],
            excluded_llm_metadata_keys=[],
            excluded_embed_metadata_keys=["parent_asin", "average_rating", "rating_number", "price", "images"],
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            metadata=review))

# Initialize Chroma DB and create/fetch collection
db = chromadb.PersistentClient(path="./chroma_db")
giftcards_collection = db.get_or_create_collection("giftcards")
vector_store = ChromaVectorStore(chroma_collection=giftcards_collection)

# ----- ADD RECORDS ----- #

# parser = SentenceSplitter(chunk_size=5000)
# nodes = parser.get_nodes_from_documents(documents, show_progress=True)

# for node in tqdm.tqdm(nodes):
#     node_embedding = embed_model.get_text_embedding(
#         node.get_content(metadata_mode=MetadataMode.EMBED)
#     )
#     node.embedding = node_embedding

# vector_store.add(nodes)

# ----- QUERY DB ----- #

def instantiate_db():
    # Initialize Chroma DB and create/fetch collection
    db = chromadb.PersistentClient(path="./chroma_db")
    return db

def recommend_products(query: str, db = None):
    if db is None:
        db = instantiate_db()
    giftcards_collection = db.get_or_create_collection("giftcards")
    vector_store = ChromaVectorStore(chroma_collection=giftcards_collection)
    template = "You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. Based on the provided dataset of products and reviews, recommend the top 3 products for a given query. Return only the ASIN of the recommended products, separated by a comma. The user query is as follows {query}.".format(query=query)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query(template)
    return response.response

def get_product_details(asin: str, db = None):
    # Get product metadata from product ASIN
    if db is None:
        db = instantiate_db()
    giftcards_collection = db.get_or_create_collection("giftcards")
    product = giftcards_collection.get(where={"parent_asin": asin}, limit=1)
    return product["metadatas"]

def justify_recommendation(query: str, product: dict):
    # Use the LLM to justify the recommendation with a list of pros, cons and neutral features
    base_template = "You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. The user has provided the following query: {query}. You have recommended the following product: {product_data}.".format(query=query, product_data=product)
    summary_template = base_template + " Based on the user query and all the information about the product, provide a newline-separated list of the reasons why you recommended this product, focusing on how it matches the user query. Structure your response as follows: 'I recommended this product for this query because: [REASONS].'"
    positive_template = base_template + " Based on the user query and all the information about the product, provide a newline-separated list of the positive features of the product. Remove any call to action e.g., 'see below' or 'learn more'."
    negative_template = base_template + " Based on the user query and all the information about the product, provide a newline-separated list of the negative features of the product. Remove any call to action e.g., 'see below' or 'learn more'."
    neutral_template = base_template + " Based on the user query and all the information about the product, provide a newline-separated list of the neutral features of the product. Remove any call to action e.g., 'see below' or 'learn more'."
    
    summary_response = OpenAI().complete(summary_template).text
    positive_response = OpenAI().complete(positive_template).text
    negative_response = OpenAI().complete(negative_template).text
    neutral_response = OpenAI().complete(neutral_template).text

    return summary_response, positive_response, negative_response, neutral_response

"""
STEPS:
1. You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. Based on the provided dataset of products and reviews, recommend the top 3 products for a given query. Return only the ASIN of the recommended products, separated by a comma. The user query is as follows {}.
2. You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. The user has provided the following query: {}. You have recommended the following product: {}. Based on the user query and all the information about the product, provided a bullet-point list of the reasons why you recommended this product. Structure your response as follows: "I recommended this product for this query because: {reasons}. The product has the following additional positive features: {features}. The product has the following additional negative features: {features}. The product has the following additional neutral features: {features}."
3. Repeat step 2 for the top 3 recommended products.
"""

if __name__ == "__main__":
    query = "gift card for a friend"
    response = recommend_products(query).split(",")
    print(response)
    details = get_product_details(response[0])
    print(details)
    justification = justify_recommendation(query, details[0])
    print(justification)
