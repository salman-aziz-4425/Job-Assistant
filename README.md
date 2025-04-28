# Job Assistant

A semantic search tool for finding relevant jobs in a MongoDB database using OpenAI embeddings.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                       Job Assistant Architecture                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  User Interface │◄────►│  Job Assistant  │◄────►│  Embedding      │
│  (CLI)          │      │  Core           │      │  Cache          │
│                 │      │                 │      │                 │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
                                  │
                                  │
         ┌─────────────────────┐  │  ┌─────────────────────────┐
         │                     │  │  │                         │
         │  OpenAI API         │◄─┴─►│  MongoDB                │
         │  - Embeddings       │     │  - Jobs Collection      │
         │  - Query Enrichment │     │  - Trades Collection    │
         │                     │     │                         │
         └─────────────────────┘     └─────────────────────────┘
```

## Features

- Semantic similarity search using OpenAI embeddings
- Query enrichment using GPT-3.5 Turbo
- Caching mechanism for embeddings to reduce API calls
- MongoDB integration for job and trade data

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install pymongo openai python-dotenv numpy tqdm
   ```
3. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MONGO_URI=your_mongodb_connection_string
   DB_NAME=your_database_name
   MAX_RESULTS=10
   ```

## Usage

### Running the Application

To use the job assistant to find relevant jobs:

```bash
python main.py
```

You will be prompted to enter a customer problem description. The system will find the most relevant jobs based on semantic similarity.

### Refreshing the Embedding Cache

To refresh the embedding cache for all jobs in the database:

```bash
python main.py --refresh-cache
```

### Running Tests

To test the functionality of the JobAssistant class:

```bash
python test_job_assistant.py
```

## Code Structure

- `JobAssistant.py`: Main class with all functionality
- `main.py`: Command-line interface
- `test_job_assistant.py`: Simple test script

## How It Works

1. The system enriches user queries using OpenAI's GPT-3.5 model to extract key components
2. Job descriptions are converted to vector embeddings using OpenAI's embedding API
3. Cosine similarity is calculated between the query embedding and job embeddings
4. Results are sorted by similarity score and returned to the user

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