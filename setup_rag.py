from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import json

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name and dimension
index_name = "rag-prof-new"
dimension = 1536

# Create Pinecone index if it doesn't exist
try:
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
except Exception as e:
    print(f"Error creating index: {e}")

# Load the review data
with open("reviews.json") as f:
    data = json.load(f)

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Process reviews and create embeddings
processed_data = []
for university, reviews in data.items():
    for review in reviews:
        # Create embedding using OpenAI API
        response = client.embeddings.create(
            input=review['review'], model="text-embedding-3-small"
        )
        print("Embedding response:", response)
        # Correctly access the embedding
        embedding = response.data[0].embedding
        processed_data.append(
            {
                "values": embedding,
                "id": f"{university} - {review['professor']}",
                "metadata": {
                    "review": review["review"],
                    "subject": review["subject"],
                    "stars": review["stars"],
                    "university": university
                }
            }
        )

# Insert the embeddings into the Pinecone index
try:
    index = pc.Index(index_name)
    upsert_response = index.upsert(
        vectors=processed_data,
        namespace="ns1"
    )
    print(f"Upserted count: {upsert_response['upserted_count']}")
except Exception as e:
    print(f"Error upserting data: {e}")

# Print index statistics
try:
    stats = index.describe_index_stats()
    print("Index statistics:", stats)
except Exception as e:
    print(f"Error describing index stats: {e}")