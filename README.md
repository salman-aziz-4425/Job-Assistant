# Job Assistant with Advanced RAG

This application uses Retrieval-Augmented Generation (RAG) to match customer problem descriptions with relevant jobs from a MongoDB database. It leverages OpenAI embeddings for semantic similarity matching and includes advanced features like query enrichment, embedding caching, and efficient similarity calculations.

## Features

- **Embedding Caching**: Stores embeddings locally to reduce API calls and improve performance
- **Query Enrichment**: Uses OpenAI to extract key details from customer problems
- **Efficient Similarity Search**: Uses NumPy for fast vector operations
- **Progress Tracking**: Shows progress during processing with tqdm
- **Detailed Job Context**: Includes comprehensive job information for better matching

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
   Or create a `.env` file based on `env.example`

## Usage

### Regular search:
```
python main.py
```
When prompted, enter a description of the customer's problem.

### Refresh embedding cache:
```
python main.py --refresh-cache
```
This will pre-calculate and cache embeddings for all jobs in the database.

## How It Works

1. The application connects to your MongoDB database to retrieve job information
2. It enriches the customer query to extract key details using OpenAI
3. It calculates embeddings for the query and job descriptions
4. It uses cosine similarity to find the most semantically similar jobs
5. It presents the top matches, ranked by similarity score

## Advanced Features

### Embedding Cache
Embeddings are stored in a local file (`embeddings_cache.pickle`) to reduce API calls and speed up subsequent searches.

### Query Enrichment
The original customer query is enhanced with extracted key details to improve matching accuracy.

### Comprehensive Job Context
Job descriptions include all available details like trade name, service type, problems, notes, etc. for better matching.

## Database Structure

The application assumes the following MongoDB collections:
- `offeredjobs`: Contains job information including trade references
- `offeredtrades`: Contains trade information

## License

MIT 