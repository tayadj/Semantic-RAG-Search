import llama_index
import llama_index.llms.openai
import os
import pandas

from .services import OntologyProcessor, NexusProcessor


class Engine:

	def __init__(self, openai_api_key: str):

		os.environ['OPENAI_API_KEY'] = openai_api_key

		self.model = llama_index.llms.openai.OpenAI(model = 'gpt-4o-mini')
		self.ontology_processor = OntologyProcessor(self.model)
		self.nexus_processor = NexusProcessor()

	def generate_ontology(self, documents_dataframe):

		ontology = []

		for idx, record in documents_dataframe.iterrows():

			result = self.ontology_processor.process(record.text)
			ontology.append(
				{
					'metadata': {
						'id': record.id,
						'file_name': record.file_name
					},
					'ontology': result
				}
			)

		return ontology

	def generate_knowledge(self, documents_dataframe, ontology_dataframe):

		knowledge_dataframe = pandas.merge(ontology_dataframe, documents_dataframe, left_on = 'chunk_source', right_on = 'file_name', how = 'inner')

		knowledge_documents = []

		for document_text in knowledge_dataframe['text'].unique():

			knowledge_documents.append(llama_index.core.schema.Document(text = document_text))

		print(knowledge_documents)

		