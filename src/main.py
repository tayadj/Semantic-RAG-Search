import pandas

import config
import core
import data
import utils



if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database({'LOCAL_STORAGE_URL': settings.LOCAL_STORAGE_URL.get_secret_value()})
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())

	'''
	# Ontology pipeline
	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology = engine.generate_ontology(documents_dataframe)
	ontology_dataframe = utils.converters.ontologyToDataframe(ontology)
	ontology_dataframe.to_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv', index = False)
	'''

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')

	knowledge_dataframe = pandas.merge(ontology_dataframe, documents_dataframe, left_on = 'chunk_source', right_on = 'file_name', how = 'inner')

	knowledge_documents = []

	for document_text in knowledge_dataframe['text'].unique():

		knowledge_documents.append(llama_index.schema.Document(document_text))




	# utils.visualization.ontologyVisualization(settings, ontology_dataframe)
	
	