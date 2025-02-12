import sys
from pprint import pprint
from typing import List

import weaviate
import weaviate.classes.query as wq
from langchain_openai.embeddings import OpenAIEmbeddings

from settings import settings

# Initialize LangChain's OpenAI embeddings
openai_embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)


# Define a function to get embeddings from OpenAI using LangChain's OpenAIEmbeddings
def vectorize_openai(text: str) -> List[float]:
    return openai_embeddings.embed_documents(text)[0]


def main():
    query_text = sys.argv[1]

    with weaviate.connect_to_local(
        headers={"X-OpenAI-Api-Key": settings.openai_api_key}
    ) as client:
        # Get the collection (ensure you have the correct collection name)
        document_chunk = client.collections.get("DocumentChunk")
        # Obtain the query vector from OpenAI's Embedding API using LangChain
        query_vector = vectorize_openai(query_text)
        # Perform the hybrid query: combine BM25 and vector search
        response = document_chunk.query.near_vector(
            near_vector=query_vector,  # For vector part of the hybrid search
            # limit=5,  # Limit the number of results
            return_metadata=wq.MetadataQuery(
                score=True
            ),  # Include the score in the response
        )

        # Inspect the response
        for o in response.objects:
            pprint(o)


if __name__ == "__main__":
    main()
