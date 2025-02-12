import re

import pypdf
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase
from text.loaders import DirectoryMultiFileLoader, DocumentLoader
from transformers import pipeline

from settings import settings

device = torch.device("cuda")

# 1. load
# 2. preprocess the text
# 3. extract the triplets
# 4. clean up the triplets
# 5. add triplets to Neo4j

triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
    device=device,
)


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = [page.extract_text() for page in reader.pages]
        metadata = reader.metadata

    return "".join(text), metadata


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def create_triplet_in_neo4j(driver, head, relationship, tail):
    """
    Creates a triplet (nodes and relationship) in Neo4j.

    Args:
        driver: The Neo4j driver instance.
        head: The head (source) entity of the triplet.
        relationship: The type of relationship between the entities.
        tail: The tail (target) entity of the triplet.
    """
    # Sanitize the relationship string: replace spaces with underscores and convert to uppercase
    relationship_label = relationship.replace(" ", "_").upper()
    source_label = head.replace(
        " ", "_"
    ).upper()  # You could customize this further if needed
    target_label = tail.replace(" ", "_").upper()

    query = f"""
    MERGE (source:{source_label} {{name: $source}})
    MERGE (target:{target_label} {{name: $target}})
    MERGE (source)-[:{relationship_label}]->(target)
    """

    # Execute the query within a write transaction
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(query, source=head, target=tail))


def clean_string(input_string: str) -> str:
    """
    Remove special characters like '-', ':', etc., from the string.
    You can modify the regex to remove any other characters you want.
    """
    # Remove non-alphanumeric characters (keeping spaces, underscores, and alphanumeric characters)
    cleaned_string = re.sub(r"[^a-zA-Z]", "_", input_string)
    return cleaned_string


def clean_triplets_grouped(triplets):
    cleaned_triplets = []

    for triplet_list in triplets:  # Preserve the outer list structure
        cleaned_group = []
        for triplet in triplet_list:
            head = triplet.get("head", "").strip()
            relationship = triplet.get("type", "").strip()
            tail = triplet.get("tail", "").strip()

            # Remove artifacts like `<subj>` or `<triplet>`
            head = clean_string(re.sub(r"<.*?>", "", head))
            relationship = clean_string(re.sub(r"<.*?>", "", relationship))
            tail = clean_string(re.sub(r"<.*?>", "", tail))

            # Skip triplets where head, relationship, or tail are empty
            if head and relationship and tail:
                cleaned_group.append({"head": head, "type": relationship, "tail": tail})

        cleaned_triplets.append(cleaned_group)  # Append the cleaned group

    return cleaned_triplets


def load_triplets_into_neo4j(driver, triplets):
    """
    Load a list of nested triplet lists into the Neo4j database.

    Args:
        driver: The Neo4j driver instance.
        triplets: A nested list of dictionaries, where each inner list contains 'head', 'type', and 'tail'.
    """
    for triplet_list in triplets:
        # Traverse the inner list (each containing triplets as dictionaries)
        for triplet in triplet_list:
            head = triplet["head"]
            relationship = triplet["type"]
            tail = triplet["tail"]

            # Skip triplets with an empty 'tail'
            if not tail:
                print(f"Skipping triplet with missing tail: {triplet}")
                continue

            # Create the triplet in Neo4j
            create_triplet_in_neo4j(driver, head, relationship, tail)


def extract_and_load_triplets_to_neo4j(text):
    print("Splitting text")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    print("Splitting documents")
    # text_splitter = SemanticChunker(
    #     embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")
    # )
    documents = text_splitter.split_documents(text)
    print("Chunking text")
    chunked_text = [doc.page_content for doc in documents]

    print("Creating batches")
    batches = [
        triplet_extractor(chunk, return_tensors=True, return_text=True)[0][
            "generated_token_ids"
        ]
        for chunk in chunked_text
    ]
    print("Extract triplets")
    extracted_text = triplet_extractor.tokenizer.batch_decode(batches)
    extracted_triplets = [extract_triplets(et) for et in extracted_text]
    print("Clean triplets")
    cleaned_triplets = clean_triplets_grouped(extracted_triplets)

    uri = settings.neo4j_uri
    auth = settings.neo4j_username, settings.neo4j_password
    driver = GraphDatabase.driver(uri, auth=auth)
    print("Loading triplets into database")
    load_triplets_into_neo4j(driver, cleaned_triplets)


if __name__ == "__main__":
    texts = [
        DocumentLoader.load_documents("documents/SWEBOKv3_chapter2.pdf"),
        DirectoryMultiFileLoader.load_documents(
            "documents/P1L2 Text Browser Exercise (Analysis) Subtitles/"
        ),
        DirectoryMultiFileLoader.load_documents(
            "documents/P1L3 Design Concepts Subtitles/"
        ),
        DirectoryMultiFileLoader.load_documents(
            "documents/P2L1 Review of UML Subtitles/"
        ),
    ]

    for i, text in enumerate(texts, start=1):
        print("Text", i)
        print("=" * 32)
        extract_and_load_triplets_to_neo4j(text)
        print()
