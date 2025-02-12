import logging
import os
from typing import Generator, List

import click
import ollama

from settings import settings

# import weaviate
# import weaviate.classes.config as wc
# from text.loaders import DirectoryMultiFileLoader, DocumentLoader
# from text.utils import convert_pdf_date_to_rfc3339
# from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
