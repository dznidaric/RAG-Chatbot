import os

from rag_agent import RagAgent

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    rag_system = RagAgent()
    rag_system.load_or_process_data()
    rag_system.initialize_qa_chain()

    print("Document Retrieval QA System (type 'exit' to quit)")
    while True:
        try:
            user_query = input("\nQuestion: ")
            if user_query.lower() in ("exit", "quit"):
                print("Exiting system...")
                break

            if not user_query.strip():
                print("Please enter a valid question")
                continue

            answer = rag_system.qa_system(user_query)
            print("Answer:", answer)

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")



