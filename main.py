import sys
from JobAssistant import JobAssistant

def main():
    job_assistant = JobAssistant()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh-cache":
        job_assistant.refresh_embeddings_cache()
        return
        
    print("Enter the customer's problem description:")
    customer_problem = input("> ")
    
    print(f"\nSearching for relevant jobs for: {customer_problem}")
    
    relevant_jobs = job_assistant.find_relevant_jobs(customer_problem)
    job_assistant.display_relevant_jobs(relevant_jobs)
    job_assistant.save_embedding_cache()

if __name__ == "__main__":
    main()









