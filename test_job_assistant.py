from JobAssistant import JobAssistant

def test_job_assistant():
    """Simple test to verify JobAssistant basic functionality."""
    print("Testing JobAssistant initialization...")
    assistant = JobAssistant()
    

    print("\nTesting embedding cache...")
    test_text = "AC unit not cooling properly"
    embedding = assistant.get_embedding(test_text)
    if embedding:
        print("Successfully generated embedding")
        cached_embedding = assistant.get_embedding(test_text)
        if cached_embedding:
            print("Successfully retrieved embedding from cache")

    print("\nTesting database connection...")
    jobs = assistant.get_all_jobs()
    if jobs:
        print(f"Successfully retrieved {len(jobs)} jobs")

        if len(jobs) > 0:
            print("\nTesting job description generation...")
            description = assistant.get_job_description(jobs[0])
            print(f"Sample job description: {description[:100]}...")
    else:
        print("No jobs found or database connection issue")
        
    print("\nTest completed!")

if __name__ == "__main__":
    test_job_assistant() 