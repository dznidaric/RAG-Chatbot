from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataProcess:
    def load_and_process_data(self, dataset_path, embedding_model):
        print("Processing and saving the documents in ChromaDB...")
        loader = DirectoryLoader(dataset_path, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=80, add_start_index=True
        )

        split_documents = text_splitter.split_documents(documents)

        vector_db = Chroma(
            embedding_function=embedding_model,
            persist_directory="chroma_db",
        )
        for doc in split_documents:
            vector_db.add_texts(texts=[doc.page_content])

        print(
            f"Successfully processed {len(split_documents)} document chunks into ChromaDB."
        )

        return vector_db
