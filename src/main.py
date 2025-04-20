import llama_index
import pandas

import config
import core
import data
import utils



def ontology_pipeline(settings, database, engine):

	# Implement multimodality

	documents = database.local_connector.load()

	image_documents = [document for document in documents if isinstance(document, llama_index.core.schema.ImageDocument)]
	described_image_documents = engine.generate_description(image_documents)

	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology = engine.generate_ontology(documents_dataframe)
	ontology_dataframe = utils.converters.ontologyToDataframe(ontology)
	ontology_dataframe.to_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv', index = False)

	utils.visualization.ontologyVisualization(settings, ontology_dataframe)

def knowledge_pipeline(settings, database, engine):

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')

	knowledge_index = engine.generate_knowledge(documents_dataframe, ontology_dataframe)

def inference_pipeline(settings, database, engine):

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')

	knowledge_index = engine.generate_knowledge(documents_dataframe, ontology_dataframe)

	query_engine = knowledge_index.as_query_engine(include_text = True, response_mode = 'tree_summarize')
	query_text = "Tell me more about American society and its influence on the cultural landscape."
	response = query_engine.query(query_text)
	print("LLM Response:\n", response)



if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database(local_storage_url = settings.LOCAL_STORAGE_URL.get_secret_value())
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())
	