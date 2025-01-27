from typing import List
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
class LlamaCppEmbeddings_(LlamaCppEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text)[0] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)[0]
        return list(map(float, embedding))

llama_model_path = "/Users/kgdhatchi/Downloads/Llama-3.2-1B-Instruct-Q4_K_M.gguf"  # Update the actual path
llama_embedder = LlamaCppEmbeddings_(model_path=llama_model_path, n_ctx=1024)
vectordb = FAISS.load_local('/Users/kgdhatchi/Desktop/11711_RAG_A2-1/src/vectorstore/', llama_embedder, allow_dangerous_deserialization=True)

num_vectors = vectordb.index.ntotal

# Display the first 3 vectors and their associated metadata (if any)
if num_vectors > 0:
    # Get the first 3 vector embeddings
    vectors = vectordb.index.reconstruct_n(0, min(3, num_vectors))
    
    # If metadata is stored separately, access it
    if hasattr(vectordb, 'docs') and vectordb.docs:
        for i, vector in enumerate(vectors):
            print(f"Vector {i+1}:")
            print(f"Embedding: {vector}")
            print(f"Metadata: {vectordb.docs[i]}")
    else:
        for i, vector in enumerate(vectors):
            print(f"Vector {i+1}:")
            print(f"Embedding: {vector}")
else:
    print("No vectors found in the vector store.")
