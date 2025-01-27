import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

if __name__ == '__main__':
    directory_path = "../../data/scraped_data"
    docs = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(root, file)
                loader = TextLoader(file_path)
                docs.extend(loader.load())  # Load and append documents

    # Step 2: Split documents into 2000-character chunks with 200-character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # Step 3: Initialize embedding model
    model_name = "nomic-ai/nomic-embed-text-v1"
    model_kwargs = {
        'device': 'cuda', 
        'trust_remote_code':True
    }
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction = "search_query:",
        embed_instruction = "search_document:"
    )

    # Step 4: Create FAISS vector store from the document splits and their embeddings
    for doc in all_splits:
        doc.page_content = "search_document: " + doc.page_content
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=embed_model)

    # Step 5: Save the FAISS index for later use (optional)
    faiss_index_path = "../../vector_store"
    vectorstore.save_local(faiss_index_path)
