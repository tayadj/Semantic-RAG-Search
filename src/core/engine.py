import base64
import llama_index
import llama_index.llms.openai
import os
import pandas

from .services import OntologyProcessor, VisionProcessor


class Engine:

	def __init__(self, openai_api_key: str):

		os.environ['OPENAI_API_KEY'] = openai_api_key

		self.model = llama_index.llms.openai.OpenAI(model = 'gpt-4o-mini')
		llama_index.core.Settings.llm = self.model

		self.ontology_processor = OntologyProcessor(self.model)
		self.vision_processor = VisionProcessor(self.model)

	async def generate_description(self, image_documents):

		described_image_documents = []

		for image_document in image_documents:

			image_document.image_resource.data = base64.b64encode(image_document.image_resource.path.read_bytes())
			result = await self.vision_processor.process(image_document)
			described_image_documents.append(
				result
			)

		return described_image_documents

	async def generate_ontology(self, documents_dataframe):

		ontology = []

		for idx, record in documents_dataframe.iterrows():

			result = await self.ontology_processor.process(record.text)
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

	async def generate_knowledge(self, documents_dataframe, ontology_dataframe):

		knowledge_dataframe = pandas.merge(ontology_dataframe, documents_dataframe, left_on = 'chunk_source', right_on = 'file_name', how = 'inner')
		knowledge_index = llama_index.core.indices.knowledge_graph.KnowledgeGraphIndex([], llm = self.model)
		node_parser = llama_index.core.node_parser.SimpleNodeParser()

		for index, record in knowledge_dataframe.iterrows():

			knowledge_document = llama_index.core.schema.Document(text = record['text'])
			knowledge_node = node_parser.get_nodes_from_documents([knowledge_document])[0]

			knowledge_index.upsert_triplet_and_node(
				(record['concept_1'], record['relationship'], record['concept_2']),
				knowledge_node
			)

		return knowledge_index