import llama_index
import mlflow
import pandas

import config
import core
import data
import utils



def ontology_pipeline(settings, database, engine):

	documents = database.local_connector.load()

	image_documents = [document for document in documents if isinstance(document, llama_index.core.schema.ImageDocument)]
	described_image_documents = engine.generate_description(image_documents)

	documents = [document for document in documents if not isinstance(document, llama_index.core.schema.ImageDocument)]
	documents = documents + described_image_documents

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
	query_text = "Tell me what is Alaska."
	response = query_engine.query(query_text)

	print("LLM Response:\n", response)

def knowledge_with_save_pipeline(settings, database, engine):

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')

	knowledge_index = engine.generate_knowledge(documents_dataframe, ontology_dataframe)


	with mlflow.start_run() as run:

		model = mlflow.llama_index.log_model(
			knowledge_index,
			artifact_path = 'llama_index',
			engine_type = 'query',
			registered_model_name = 'llama_index_knowledge_index'
		)
		model_uri = model.model_uri

		print(f'Model identifier for loading: {model_uri}')


if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database(local_storage_url = settings.LOCAL_STORAGE_URL.get_secret_value())
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())

	mlflow.set_experiment('Semantic-RAG-Search') 
	mlflow.set_tracking_uri(settings.MLFLOW_HOST_URI.get_secret_value())
	mlflow.llama_index.autolog()
	