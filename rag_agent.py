import os

from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from data_process import DataProcess


class RagAgent:
    def __init__(self, dataset_path="data", persist_dir="chroma_db"):
        self.embedding_model = HuggingFaceHubEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        self.vector_db = None
        self.qa_chain = None
        self.client = None
        self.retriever = None
        self.dataset_path = dataset_path
        self.persist_dir = persist_dir

    def vector_db_exists(self):
        return os.path.exists(self.persist_dir) and os.listdir(self.persist_dir)

    def load_or_process_data(self):
        if self.vector_db_exists():
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
            )
        else:
            process_class = DataProcess()
            self.vector_db = process_class.load_and_process_data(
                self.dataset_path, self.embedding_model
            )
        print("Total documents in ChromaDB:", self.vector_db._collection.count())

    def initialize_qa_chain(self):
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        self.client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")

    def qa_system(self, query):
        relevant_docs = list(self.retriever.invoke(query))
        if not relevant_docs:
            return f"No relevant documents found for query: {query}"

        context = "\n".join([doc.page_content for doc in relevant_docs])

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
        )

        return response["choices"][0]["message"]["content"]
