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
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from tqdm import tqdm
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

load_dotenv()

# ----- DATA CLEANING ----- #

# Download product products and reviews
product_reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
product_meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", trust_remote_code=True)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Convert to pandas df
product_reviews_df = pd.DataFrame(product_reviews_dataset['full'])
product_meta_df = pd.DataFrame(product_meta_dataset['full'])

# For each product, add all relevant reviews.
# Combine all reviews for a given parent ASIN, truncating text to fit chunk size.
product_reviews_df = product_reviews_df.groupby('parent_asin').agg({'text': '\n'.join}).reset_index()
product_reviews_df.loc[:, 'text'] = product_reviews_df['text'].apply(lambda x: x[:7500])
product_reviews_df = product_reviews_df.merge(product_meta_df, on='parent_asin')

# Create a new df with only the relevant columns
product_df = product_reviews_df[['parent_asin', 'text', 'title', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'details', 'subtitle']]

# Remove rows with no price data and convert to float
product_df.loc[:, "price"] = product_df["price"].apply(lambda x: x.replace("None", "0"))
product_df.loc[:, "price"] = product_df["price"].apply(lambda x: x.replace("from ", ""))
product_df.loc[:, "price"] = product_df["price"].apply(lambda x: x.replace("-", ""))
product_df.loc[:, "price"] = pd.to_numeric(product_df["price"], errors='coerce')
product_df = product_df[product_df["price"] != 0]
product_df = product_df.dropna(subset=["price", "images"])

# Convert columns to types that the LLM can handle
product_df.loc[:, "images"] = product_df["images"].apply(lambda x: x.get("hi_res", None) if isinstance(x, dict) else None)
product_df.loc[:, "images"] = product_df["images"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
product_df.loc[:, "features"] = product_df["features"].apply(lambda x: '\n'.join(x) if isinstance(x, list) else x)

# ----- LLM PREP ----- #

# Convert df to list of dicts
product_data = product_df.to_dict(orient='records')

# Pop unsupported columns from the dict.
for product in product_data:
    product.pop("description")

# Create LlamaIndex documents for each review
documents = []
for review in product_data:
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
product_collection = db.get_or_create_collection("product")
vector_store = ChromaVectorStore(chroma_collection=product_collection)

# ----- ADD RECORDS ----- #

# # UNCOMMENT TO BUILD VECTOR STORE
# parser = SentenceSplitter(chunk_size=10000)
# nodes = parser.get_nodes_from_documents(documents, show_progress=True)

# import concurrent.futures

# def process_node(node):
#     node_embedding = embed_model.get_text_embedding(
#         node.get_content(metadata_mode=MetadataMode.EMBED)
#     )
#     node.embedding = node_embedding

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(process_node, nodes), total=len(nodes)))

# vector_store.add(nodes)

# ----- QUERY DB ----- #

def instantiate_db():
    # Initialize Chroma DB and create/fetch collection
    db = chromadb.PersistentClient(path="./chroma_db")
    return db

def recommend_products(query: str, db = None, min_rating: float = 0, min_reviews: int = 100, max_price: float = 10):
    if db is None:
        db = instantiate_db()
    product_collection = db.get_or_create_collection("product")
    vector_store = ChromaVectorStore(chroma_collection=product_collection)

    # Build hardcoded filters
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="average_rating", value=min_rating, operator=FilterOperator.GTE),
            MetadataFilter(key="rating_number", value=min_reviews, operator=FilterOperator.GTE),
            MetadataFilter(key="price", value=max_price, operator=FilterOperator.LTE),
            ]
        )

    # Set up LLM filter
    template = "You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. Based on the provided dataset of products, reviews and the users query, recommend the top 3 products for a given query. Return only the ASIN of the recommended products, separated by a comma. If there is no good match, return an empty list. Overindex on the properties that the user has stated, and think deeply about how every word the user have given in the query could impact what they're looking for, assuming they have chosen each word deliberately. The user query is as follows {query}.".format(query=query)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(filters=filters)
    
    response = query_engine.query(template)
    return response.response

def get_product_details(asin: str, db = None):
    # Get product metadata from product ASIN
    if db is None:
        db = instantiate_db()
    product_collection = db.get_or_create_collection("product")
    product = product_collection.get(where={"parent_asin": asin}, limit=1)
    return product["metadatas"]

def justify_recommendation(query: str, product: dict):
    # Use the LLM to justify the recommendation with a list of pros, cons and neutral features
    base_template = "You are a product recommendation expert, tasked with recommending products to users based on what they say they want, and what could be good for them. Be as concise as possible, limiting to around 30 characters. Do not include any call to action e.g., 'see below' or 'learn more'. I have provided the following query: {query}. You have recommended the following product: {product_data}.".format(query=query, product_data=product)
    summary_template = base_template + " Based on my query and all the information about the product, provide a newline-separated list of the reasons why you recommended this product, focusing on how it matches my query."
    positive_template = base_template + " Based on my query and all the information about the product, provide a newline-separated list of the positive features of the product, prioritising those that best serve my original query. Thinky deeply about why these features are important to me and whether they are truly beneficial based on the query."
    negative_template = base_template + " Based on my query and all the information about the product, provide a newline-separated list of the negative features of the product which may prevent me from purchasing this product. Think deeply about why these features could impact me based on the query."
    neutral_template = base_template + " Based on my query and all the information about the product, provide a newline-separated list of the neutral features of the product. These features are useful to know but may not necessarily be a deciding factor my purchasing decision. Think deeply about why these features could impact me based on the query."
    
    summary_response = OpenAI().complete(summary_template).text
    positive_response = OpenAI().complete(positive_template).text
    negative_response = OpenAI().complete(negative_template).text
    neutral_response = OpenAI().complete(neutral_template).text

    return summary_response, positive_response, negative_response, neutral_response

if __name__ == "__main__":
    query = "anything!"
    response = recommend_products(query).split(",")
    print(response)
    details = get_product_details(response[0])
    if details:
        print(details)
        justification = justify_recommendation(query, details[0])
        print(justification)
    else:
        print("No product match")        
