import argparse
import logging
from typing import List

import genanki
import weaviate
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langsmith import traceable
from pydantic import BaseModel, Field
from text.utils import gen_rand_id

from settings import settings

load_dotenv()
set_debug(True)


log_file = "app.log"
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the logging level as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Keep logging to the console
        logging.FileHandler(log_file),  # Add logging to a file
    ],
)

system = """You are an assistant that generates multiple-choice questions for educational
purposes. Given the following chunk of text, create **{num_questions} multiple-choice questions** that test
the reader's understanding of the content. You shall retrieve the data from the database in
order to get the text to generate the cards from. Use the WeaviateQueryTool tool to query
the database. Ensure that when calling the tool, you use the `subject` as the argument to
the WeaviateQueryTool tool.

For each question:
1. Provide one correct answer.
2. Provide three plausible but incorrect options (distractors) to challenge the reader.
3. Ensure the correct answer is accurate and based on the text.
4. Avoid vague or overly broad questions.
5. Do not reference "the text,""the subject," or even the document's name in the question.
Frame questions naturally as if this were a question in a quiz.

Requirements:
- Provide a mix of question difficulties, from easy to challenging, as appropriate for
  graduate-level students in a top-tier computer science program.
- Include different types of questions, such as:
  - Vocabulary (e.g., definitions or terminology)
  - Conceptual understanding
  - Application of ideas or techniques
  - Critical thinking and analysis
- Use precise language and avoid repetition in question structure.

Guidelines:
1. Avoid mentioning the text, subject, or example explicitly in the question.
2. Focus on the concepts, definitions, techniques, and ideas presented.
3. Do not generate questions that rely on specific examples, illustrations, or graphics unless
   the question includes sufficient context to answer independently.
4. Provide a mix of question difficulties appropriate for a graduate-level course at a top 10 public university.

Output format:
Return a Python list of dictionaries. Each dictionary should have the following keys:
    - "question": A string representing the question.
    - "options": A list of four strings, representing the multiple-choice options.
    - "answer": A string that matches one of the options and represents the correct answer.

Example output:
[
    {{
    "question": "What is the primary purpose of machine learning?",
    "options": ["To write code", "To perform statistical analysis", "To make predictions from data", "To replace human workers"],
    "answer": "To make predictions from data"
    }},
    {{
    "question": "Which of the following best describes supervised learning?",
    "options": ["Training a model with labeled data", "Using neural networks for image recognition", "Clustering data points into groups", "Applying reinforcement learning principles"],
    "answer": "Training a model with labeled data"
    }},
    {{
    "question": "What would likely happen if a dataset contains significant class imbalance?",
    "options": ["The model may underfit the majority class", "The model may ignore the minority class", "The model's accuracy will always increase", "The model will become resistant to overfitting"],
    "answer": "The model may ignore the minority class"
    }},
    ...
]

Do not stray from the format given in the Example output above. No opinions or any text, only the
JSON output shall be returned.

Make sure the questions are varied in scope and difficulty, and use graduate-level reasoning and depth
and use the given context and subject to generate the questions.

Context: {context}

Subject: {subject}
"""


mcq_model = genanki.Model(
    model_id=gen_rand_id(),
    name="Multiple Choice Model",
    fields=[
        {"name": "Question"},
        {"name": "Option1"},
        {"name": "Option2"},
        {"name": "Option3"},
        {"name": "Option4"},
        {"name": "Answer"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Question}}<br><br>"
            '<label><input type="radio" name="options" value="{{Option1}}">{{Option1}}</label><br>'
            '<label><input type="radio" name="options" value="{{Option2}}">{{Option2}}</label><br>'
            '<label><input type="radio" name="options" value="{{Option3}}">{{Option3}}</label><br>'
            '<label><input type="radio" name="options" value="{{Option4}}">{{Option4}}</label><br>',
            "afmt": '{{FrontSide}}<hr id="answer">{{Answer}}',
        },
    ],
)


class AnkiCard(BaseModel):
    """Simplified model for an Anki card."""

    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of multiple-choice options")
    answer: str = Field(description="The correct answer")


class AnkiDeck(BaseModel):
    deck: List[AnkiCard] = Field(description="A deck of Anki cards")


openai_embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)


# Define a function to get embeddings from OpenAI using LangChain's OpenAIEmbeddings
def vectorize_openai(text: str) -> List[float]:
    return openai_embeddings.embed_documents(text)[0]


# class WeaviateQueryTool:
#     @tool(parse_docstring=True)
#     def run(self, subject: str):
#         """
#         Retrieves relevant text from the Weaviate database by performing a hybrid search using
#         both BM25 (traditional keyword search) and vector-based search. The method leverages
#         OpenAI's Embedding API for vectorization of the subject, and queries Weaviate's "DocumentChunk"
#         collection for the most relevant documents.

#         Args:
#             subject (str): The query string or subject for which to retrieve relevant information from Weaviate.

#         Returns:
#             List[dict]: A list of documents (or document chunks) that match the hybrid search query,
#                         including metadata like the relevance score.

#         Raises:
#             ConnectionError: If there's an issue connecting to the Weaviate instance.
#             QueryExecutionError: If the query execution fails.

#         Example:
#             response = WeaviateQueryTool.run("Python programming")
#             print(response)

#         This method performs a hybrid query that combines the BM25 traditional keyword search and
#         the vector search using embeddings to retrieve the most relevant documents stored in Weaviate's
#         "DocumentChunk" collection. The results include metadata such as relevance scores to help rank
#         the returned documents based on their relevance to the subject.
#         """
#         print("In the WeaviateQueryTool.run method")
#         logging.info(f"Calling WeaviateQueryTool with args: {subject}")
#         with weaviate.connect_to_local(
#             headers={"X-OpenAI-Api-Key": settings.openai_api_key}
#         ) as client:  # Or use your Weaviate instance
#             # Get the collection (ensure you have the correct collection name)
#             db = Weaviate(client, index_name="DocumentChunk", text_key="text")
#             retriever = db.as_retriever(search_type="mmr")

#         return response.objects


def create_anki_note(card: AnkiCard, model: genanki.Model = mcq_model) -> genanki.Note:
    return genanki.Note(
        model=model,
        fields=[
            card.question,
            *card.options,
            card.answer,
        ],
    )


# def create_agent(model: str = "gpt-4"):
#     llm = ChatOpenAI(
#         model=model,
#         api_key=settings.openai_api_key,
#         verbose=True,
#         # temperature=0,
#     )
#     tools = [
#         Tool(
#             name="WeaviateQueryTool",
#             func=WeaviateQueryTool().run,
#             description=(
#                 "Query Weaviate to retrieve the relevant "
#                 "document chunks for a specific subject and keywords. "
#                 "The tool is called in the following manner: "
#                 "`WeaviateQueryTool.run(subject)` "
#                 "where `subject` is the user given subject"
#             ),
#         )
#     ]
#     return initialize_agent(
#         tools=tools,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         llm=llm,
#         verbose=True,
#     )


@traceable
def generate_responses(
    subject: str, num_questions: int = 20, model: str = "gpt-4"
) -> List[dict]:
    """Generates questions and answers for the given subject."""
    system_prompt = system.replace("{num_questions}", str(num_questions))
    llm = ChatOpenAI(model=model, api_key=settings.openai_api_key, verbose=True)
    # llm = OpenAI(
    #     model=model,
    #     api_key=settings.openai_api_key,
    #     verbose=True,
    #     temperature=0,
    # )
    # llm.bind_tools([WeaviateQueryTool.run])

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{subject}")]
    )
    parser = PydanticOutputParser(pydantic_object=List[AnkiCard])
    # chain = prompt | agent
    with weaviate.connect_to_local(
        headers={"X-OpenAI-Api-Key": settings.openai_api_key}
    ) as client:
        # vectorstore = Weaviate(
        #     client=client,
        #     embedding_function=openai_embeddings.embed_query
        #     index_name="your_index",  # Replace with your Weaviate index name
        #     text_key="text",  # Adjust based on your schema
        # )
        db = WeaviateVectorStore(
            client=client,
            index_name="DocumentChunk",
            text_key="text",
            embedding=openai_embeddings,
        )
        retriever = db.as_retriever(search_type="similarity")
        # chain = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm, chain_type="stuff", retriever=retriever
        # )
        # responses = chain.invoke({"subject": subject})
        rag_chain = (
            {"context": retriever, "subject": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
        )
        responses = rag_chain.invoke(subject)
    return responses


# Save to Anki Format
def save_to_anki_format(cards: List[dict], output_file: str, deckname: str):
    """Saves questions and answers in Anki's importable TSV format."""
    deck = genanki.Deck(deck_id=gen_rand_id(), name=deckname)
    for card in cards:
        deck.add_note(create_anki_note(AnkiCard(**card), model=mcq_model))

    genanki.Package(deck).write_to_file(output_file)


# Main MVP Function
def main():
    parser = argparse.ArgumentParser(
        description="Generate Anki cards from text documents."
    )
    parser.add_argument(
        "-s", "--subject", help="The subject to make the Anki cards about"
    )
    parser.add_argument(
        "-n",
        "--deck-name",
        default="My Deck",
        help="Name of the Anki deck (default: 'My Deck').",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="output.apkg",
        help="Path to save the output Anki deck (default: 'output.apkg').",
    )

    parser.add_argument(
        "-q",
        "--questions",
        default=20,
        help="The number of questions to generate.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Show logging statements",
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Name of the LLM model to use (default: gpt-4).",
    )

    args = parser.parse_args()

    logging.info("Generating responses...")
    responses = generate_responses(
        subject=args.subject, model=args.model, num_questions=args.questions
    )
    logging.info("Saving deck...")
    save_to_anki_format(responses, args.output_file, args.deck_name)
    logging.info(f"Anki deck saved to {args.output_file}!")


# Example Usage
if __name__ == "__main__":
    main()
