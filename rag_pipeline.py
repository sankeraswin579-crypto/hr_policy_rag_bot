import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from groq import Groq

from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
    TOP_K
)


class RAGPipeline:

    def __init__(self):

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.index = None
        self.chunks = []

    def build_vectorstore(self, chunks):

        self.chunks = chunks

        embeddings = self.embedder.encode(
            chunks,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

    def retrieve(self,
                 query,
                 k=TOP_K):

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            k
        )

        retrieved_chunks = []

        for idx in indices[0]:

            if idx < len(self.chunks):
                retrieved_chunks.append(
                    self.chunks[idx]
                )

        return retrieved_chunks

    def answer(self, question):

        docs = self.retrieve(question)

        context = "\n\n".join(docs)

        prompt = f"""
You are an HR Policy Assistant.

Answer ONLY using the context.

If the answer is not found,
say:
'Information not found in uploaded documents.'

Context:
{context}

Question:
{question}
"""

        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        answer = completion.choices[0].message.content

        return answer, docs