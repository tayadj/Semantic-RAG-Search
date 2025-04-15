import llama_index
import llama_index.llms.openai
import os

from .services import OntologyProcessor


class Engine:

	def __init__(self, openai_api_key: str):

		os.environ['OPENAI_API_KEY'] = openai_api_key

		self.model = llama_index.llms.openai.OpenAI(model = 'gpt-4o-mini')
		self.ontology_processor = OntologyProcessor(self.model)

	def generate_ontology(self, dataframe):

		pass