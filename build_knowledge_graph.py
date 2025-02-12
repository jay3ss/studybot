import math

import pypdf
import torch
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from text.loaders import EnhancedPDFLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from settings import settings

model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")

llm = ChatOllama(model="llama3.2", temperature=0)
llm_transformer = LLMGraphTransformer(
    llm=llm,
    # node_properties=["id", "title"]
)

print("Loading documents")
documents = EnhancedPDFLoader("documents/SWEBOKv3_chapter2.pdf").load()
print("Converting documents")
# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# print(f"Nodes: {graph_documents[0].nodes}")
# print(f"Nodes: {graph_documents[0].relationships}")

graph = Neo4jGraph(
    url=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password,
    refresh_schema=False,
)
print("Adding documents to graph")
# graph.add_graph_documents(graph_documents, include_source=True)
# graph.add_graph_documents(graph_documents, baseEntityLabel=True)
print("Done!")


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = [page.extract_text() for page in reader.pages]
        metadata = reader.metadata

    return text, metadata


def from_text_to_neo4j_graph(graph, text, span_length=128, verbose=False):
    """
    Processes text into spans, extracts relations using the REBEL model,
    and populates a Neo4j graph.

    Parameters:
    - graph: Neo4jGraph instance for interacting with the Neo4j database.
    - text: The input text to process.
    - span_length: The maximum number of tokens per span.
    - verbose: Whether to print detailed logs.
    """
    # Tokenize the entire text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append(
            [start + span_length * i, start + span_length * (i + 1)]
        )
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # Transform input into spans
    tensor_ids = [
        inputs["input_ids"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries
    ]
    tensor_masks = [
        inputs["attention_mask"][0][boundary[0] : boundary[1]]
        for boundary in spans_boundaries
    ]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks),
    }

    # Generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences,
    }
    generated_tokens = model.generate(**inputs, **gen_kwargs)

    # Decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # Populate the Neo4j graph
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            # relation["meta"] = {"spans": [spans_boundaries[current_span_index]]}
            meta = {
                "span_start": spans_boundaries[current_span_index][0],
                "span_end": spans_boundaries[current_span_index][1],
            }

            # Extract subject, relation, and object
            head = relation["head"]
            type_ = relation["type"]
            tail = relation["tail"]

            # Add entities and relationship to the graph
            try:
                graph.query(
                    "MERGE (e:Entity {name: $name}) RETURN e",
                    params={"name": head.strip()},
                )
                graph.query(
                    "MERGE (e:Entity {name: $name}) RETURN e",
                    params={"name": tail.strip()},
                )
                graph.query(
                    """
                    MATCH (e1:Entity {name: $head}), (e2:Entity {name: $tail})
                    MERGE (e1)-[r:RELATIONSHIP {type: $type}]->(e2)
                    RETURN r
                    """,
                    params={
                        "head": head.strip(),
                        "tail": tail.strip(),
                        "type": type_.strip(),
                        # "meta": meta,
                    },
                )
            except Exception as e:
                if verbose:
                    print(f"Error adding relation {relation}: {e}")

        i += 1

    print("Graph successfully populated!")


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                relations.append(
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
                relations.append(
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
        relations.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return relations


if __name__ == "__main__":
    print("Loading documents")
    pdf_path = "documents/SWEBOKv3_chapter2.pdf"
    text, metadata = extract_text_from_pdf(pdf_path)
    print("Converting documents")
    # graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # print(f"Nodes: {graph_documents[0].nodes}")
    # print(f"Nodes: {graph_documents[0].relationships}")

    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        refresh_schema=False,
    )
    print("Finding entities and relationships")
    from_text_to_neo4j_graph(graph, text, verbose=True)
    # graph.add_graph_documents(graph_documents, include_source=True)
    # graph.add_graph_documents(graph_documents, baseEntityLabel=True)
    print("Done!")
