import argparse
import logging
from typing import List

import genanki
import weaviate
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langsmith import traceable
from pydantic import BaseModel, Field
from text.utils import gen_rand_id, get_deck_id
from weaviate.classes.query import Filter, Sort

from settings import settings

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
the reader's understanding of the content.

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
- Don't write this as code, it's not. Just write the questions in the given format

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


def create_anki_note(card: AnkiCard, model: genanki.Model = mcq_model) -> genanki.Note:
    return genanki.Note(
        model=model,
        fields=[
            card.question,
            *card.options,
            card.answer,
        ],
    )


@traceable
def generate_responses(
    subject: str, num_questions: int = 20, model: str = "deepseek-r1:14b"
) -> List[dict]:
    """Generates questions and answers for the given subject."""
    llm = ChatOllama(model=model, verbose=True, temperature=0.1, num_ctx=32768)
    structured_llm = llm.with_structured_output(AnkiDeck, method="json_schema")

    prompt = ChatPromptTemplate.from_messages([("system", system)])
    with weaviate.connect_to_local() as client:
        collection = client.collections.get("DocumentChunk")

        def retriever(source):
            return collection.query.fetch_objects(
                filters=Filter.by_property("source").equal(source),
                sort=Sort.by_property(name="index", ascending=True),
            )

        def prepare_prompt_input(inputs):
            chunks = retriever(inputs["source"])
            context = "".join(
                [chunk.properties.get("text", "") for chunk in chunks.objects]
            )
            return {
                "context": context,
                "num_questions": num_questions,
            }

        prepared = prepare_prompt_input(
            {"source": subject, "num_questions": num_questions}
        )
        rendered_prompt = prompt.invoke(prepared)
    responses = structured_llm.invoke(rendered_prompt)
    return responses


def save_to_anki_format(cards: AnkiDeck, output_file: str, deckname: str):
    """Saves questions and answers in Anki's importable TSV format."""
    deck = genanki.Deck(deck_id=get_deck_id(deckname), name=deckname)
    for card in cards.deck:
        deck.add_note(
            genanki.Note(
                model=mcq_model, fields=[card.question] + card.options + [card.answer]
            )
        )

    genanki.Package(deck).write_to_file(output_file)


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
        default="deepseek-r1:14b",
        help="Name of the LLM model to use (default: deepseek-r1:14b).",
    )

    args = parser.parse_args()

    logging.info("Generating responses...")
    responses = generate_responses(
        subject=args.subject, model=args.model, num_questions=args.questions
    )
    logging.info("Saving deck...")
    save_to_anki_format(responses, args.output_file, args.deck_name)
    logging.info(f"Anki deck saved to {args.output_file}!")


if __name__ == "__main__":
    main()
