import pymongo
import openai
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import sys
import json
from dotenv import load_dotenv
import hashlib
import pickle
from datetime import datetime
from tqdm import tqdm


load_dotenv()

# Initialize OpenAI client with older SDK version
openai_api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
openai.api_key = openai_api_key  # The older method for setting API key

# MongoDB connection
mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
db_name = os.environ.get("DB_NAME", "autobot")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]
jobs_collection = db["offeredjobs"]
trades_collection = db["offeredtrades"]

# Application settings
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", 10))
EMBEDDING_CACHE_FILE = "embeddings_cache.pickle"

# Initialize embedding cache
embedding_cache = {}
if os.path.exists(EMBEDDING_CACHE_FILE):
    try:
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            embedding_cache = pickle.load(f)
        print(f"Loaded {len(embedding_cache)} cached embeddings")
    except Exception as e:
        print(f"Error loading embedding cache: {e}")

def save_embedding_cache():
    """Save the embedding cache to disk"""
    try:
        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(embedding_cache, f)
        print(f"Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Error saving embedding cache: {e}")

def get_embedding(text, force_refresh=False):
    if not text:
        return None
    
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    if not force_refresh and text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response['data'][0]['embedding']
    
        embedding_cache[text_hash] = embedding
        if len(embedding_cache) % 10 == 0:
            save_embedding_cache()
            
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    return dot_product / (norm_1 * norm_2) if norm_1 * norm_2 != 0 else 0

def get_all_jobs():
    """Get all jobs with trade information"""
    try:
        pipeline = [
            {"$lookup": {
                "from": "offeredtrades",
                "localField": "trade",
                "foreignField": "_id",
                "as": "trade_info"
            }}
        ]
        return list(jobs_collection.aggregate(pipeline))
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []

def get_job_description(job, include_details=True):
    trade_name = get_trade_name(job)
    service_type = job.get('service', job.get('serviceType', ''))
    job_item = job.get('jobItem', job.get('jobName', ''))
    description = f"{trade_name} {service_type} {job_item}".strip()
    if include_details:
        if 'problem' in job and job['problem']:
            description += f". Problem: {job['problem']}"
        for field in ['details', 'description', 'notes', 'comments']:
            if field in job and job[field]:
                description += f". {field.capitalize()}: {job[field]}"
    
    return description

def get_trade_name(job):
    trade_name = job.get('trade', '')
    if 'trade_info' in job and job['trade_info']:
        if isinstance(job['trade_info'], list) and len(job['trade_info']) > 0:
            trade_name = job['trade_info'][0].get('tradeName', '')
        else:
            trade_name = job['trade_info'].get('tradeName', '')
    return trade_name

def enrich_query(customer_problem):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract the key components from problem descriptions for service jobs. Include trade type, equipment details, and issue symptoms."},
                {"role": "user", "content": f"Customer problem: {customer_problem}\n\nExtract the key components from this problem description:"}
            ],
            max_tokens=150,
            temperature=0.2
        )
        enriched_query = response['choices'][0]['message']['content'].strip()
        return f"{customer_problem} {enriched_query}"
    except Exception as e:
        print(f"Error enriching query: {e}")
        return customer_problem

def find_relevant_jobs(customer_problem, all_jobs):
    print("Enriching query...")
    enriched_query = enrich_query(customer_problem)
    print(f"Enriched query: {enriched_query}")
    
    problem_embedding = get_embedding(enriched_query)
    if not problem_embedding:
        return []
    
    print("Calculating similarities...")
    job_similarities = []

    for job in tqdm(all_jobs, desc="Finding matches"):
        job_description = get_job_description(job)
        job_embedding = get_embedding(job_description)
        
        if job_embedding:
            similarity = cosine_similarity(problem_embedding, job_embedding)
            job_similarities.append((job, similarity))
    
    job_similarities.sort(key=lambda x: x[1], reverse=True)
    

    return job_similarities[:MAX_RESULTS]

def refresh_embeddings_cache():
    all_jobs = get_all_jobs()
    print(f"Refreshing embeddings for {len(all_jobs)} jobs...")
    
    for job in tqdm(all_jobs, desc="Caching embeddings"):
        job_description = get_job_description(job)
        get_embedding(job_description, force_refresh=True)
    
    save_embedding_cache()
    print("Embedding cache refresh complete!")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh-cache":
        refresh_embeddings_cache()
        return
        
    print("Enter the customer's problem description:")
    customer_problem = input("> ")
    
    print(f"\nSearching for relevant jobs for: {customer_problem}")
    
    all_jobs = get_all_jobs()
    if not all_jobs:
        print("No jobs found in the database.")
        return
    
    print(f"Found {len(all_jobs)} total jobs. Finding most relevant matches...")
    
    relevant_jobs = find_relevant_jobs(customer_problem, all_jobs)
    
    if not relevant_jobs:
        print("Could not find any relevant jobs.")
        return
    
    print(f"\nTop {len(relevant_jobs)} relevant jobs:")
    print("="*50)
    
    for i, (job, similarity) in enumerate(relevant_jobs, 1):
        job_id = job['_id']
        trade_name = get_trade_name(job)
        service_type = job.get('service', job.get('serviceType', 'N/A'))
        job_item = job.get('jobItem', job.get('jobName', 'N/A'))
        
        print(f"\n{i}. Job ID: {job_id}")
        print(f"   Trade: {trade_name}")
        print(f"   Service: {service_type}")
        print(f"   Job: {job_item}")
        print(f"   Similarity Score: {similarity:.4f}")
    

    save_embedding_cache()

if __name__ == "__main__":
    main()









