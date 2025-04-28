import os
import pickle
import hashlib
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import openai
from tqdm import tqdm

load_dotenv()

class JobAssistant:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
        self.mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = os.environ.get("DB_NAME", "jobs_database")
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client[self.db_name]
        self.jobs_collection = self.db["offeredjobs"]
        self.trades_collection = self.db["offeredtrades"]
        self.embedding_cache = {}
        self.embedding_cache_file = "embeddings_cache.pickle"
        self.MAX_RESULTS = int(os.environ.get("MAX_RESULTS", 10))
        openai.api_key = self.openai_api_key
        self.load_embedding_cache()
        
    def load_embedding_cache(self):
        """Load embeddings from cache file if it exists."""
        try:
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} embeddings from cache")
        except Exception as e:
            print(f"Error loading embedding cache: {e}")
            self.embedding_cache = {}

    def save_embedding_cache(self):
        """Save embeddings to cache file."""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving embedding cache: {e}")

    def get_embedding(self, text, force_refresh=False):
        """Get embedding for text using OpenAI's embedding API with caching."""
        if not text:
            return None
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if not force_refresh and text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response['data'][0]['embedding']
        
            self.embedding_cache[text_hash] = embedding
            if len(self.embedding_cache) % 10 == 0:
                self.save_embedding_cache()
                
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
            
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)
        return dot_product / (norm_1 * norm_2) if norm_1 * norm_2 != 0 else 0

    def get_all_jobs(self):
        """Get all jobs with trade information."""
        try:
            pipeline = [
                {"$lookup": {
                    "from": "offeredtrades",
                    "localField": "trade",
                    "foreignField": "_id",
                    "as": "trade_info"
                }}
            ]
            return list(self.jobs_collection.aggregate(pipeline))
        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return []

    def get_trade_name(self, job):
        """Extract trade name from job data."""
        trade_name = job.get('trade', '')
        if 'trade_info' in job and job['trade_info']:
            if isinstance(job['trade_info'], list) and len(job['trade_info']) > 0:
                trade_name = job['trade_info'][0].get('tradeName', '')
            else:
                trade_name = job['trade_info'].get('tradeName', '')
        return trade_name

    def get_job_description(self, job, include_details=True):
        """Create a descriptive string from job data."""
        trade_name = self.get_trade_name(job)
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

    def enrich_query(self, customer_problem):
        """Enhance the customer query with AI-extracted components."""
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

    def find_relevant_jobs(self, customer_problem):
        print("Enriching query...")
        enriched_query = self.enrich_query(customer_problem)
        print(f"Enriched query: {enriched_query}")
        
        problem_embedding = self.get_embedding(enriched_query)
        if not problem_embedding:
            return []
        
        all_jobs = self.get_all_jobs()
        if not all_jobs:
            print("No jobs found in the database.")
            return []
            
        print(f"Found {len(all_jobs)} total jobs. Finding most relevant matches...")
        print("Calculating similarities...")
        job_similarities = []

        for job in tqdm(all_jobs, desc="Finding matches"):
            job_description = self.get_job_description(job)
            job_embedding = self.get_embedding(job_description)
            
            if job_embedding:
                similarity = self.cosine_similarity(problem_embedding, job_embedding)
                job_similarities.append((job, similarity))
        
        job_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return job_similarities[:self.MAX_RESULTS]

    def refresh_embeddings_cache(self):
        """Refresh all embeddings in the cache."""
        all_jobs = self.get_all_jobs()
        print(f"Refreshing embeddings for {len(all_jobs)} jobs...")
        
        for job in tqdm(all_jobs, desc="Caching embeddings"):
            job_description = self.get_job_description(job)
            self.get_embedding(job_description, force_refresh=True)
        
        self.save_embedding_cache()
        print("Embedding cache refresh complete!")
        
    def display_relevant_jobs(self, relevant_jobs):
        """Display relevant jobs with their matching scores."""
        if not relevant_jobs:
            print("Could not find any relevant jobs.")
            return
        
        print(f"\nTop {len(relevant_jobs)} relevant jobs:")
        print("="*50)
        
        for i, (job, similarity) in enumerate(relevant_jobs, 1):
            job_id = job['_id']
            trade_name = self.get_trade_name(job)
            service_type = job.get('service', job.get('serviceType', 'N/A'))
            job_item = job.get('jobItem', job.get('jobName', 'N/A'))
            
            print(f"\n{i}. Job ID: {job_id}")
            print(f"   Trade: {trade_name}")
            print(f"   Service: {service_type}")
            print(f"   Job: {job_item}")
            print(f"   Similarity Score: {similarity:.4f}")



