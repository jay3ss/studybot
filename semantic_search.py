import sys
from pprint import pprint

import weaviate
import weaviate.classes.query as wq

from settings import settings


def main():
    query_text = sys.argv[1]

    with weaviate.connect_to_local(
        headers={"X-OpenAI-Api-Key": settings.openai_api_key}
    ) as client:  # Or use your Weaviate instance
        # Get the collection (ensure you have the correct collection name)
        document_chunk = client.collections.get("DocumentChunk")
        # Perform the hybrid query: combine BM25 and vector search
        response = document_chunk.query.near_text(
            query=query_text,  # For BM25 part of the hybrid search
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
