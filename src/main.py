import fastapi
import llama_index
import mlflow
import pandas
import pydantic
import uvicorn

import config
import core
import data
import utils

# Implement asynchronous run
# RAG Evaluation Pipeline via MLFlow

application = fastapi.FastAPI()

@application.head('/ontology')
def ontology_pipeline():

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



@application.head('/knowledge')
def knowledge_pipeline():

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents) # save as the tag	

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



class InferenceRequest(pydantic.BaseModel):

	query: str

@application.post('/inference')
def inference_pipeline(request: InferenceRequest):

	model_name = settings.MLFLOW_MODEL.get_secret_value()
	model_version = client.get_registered_model('llama_index_knowledge_index').latest_versions[0].version
	model_uri = f'models:/{model_name}/{model_version}'

	knowledge_index = mlflow.llama_index.load_model(model_uri)

	query_engine = knowledge_index.as_query_engine(include_text = True, response_mode = 'tree_summarize') # Review way of .as_query_engine
	response = query_engine.query(request.query)

	return {'response': str(response)}



if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database(local_storage_url = settings.LOCAL_STORAGE_URL.get_secret_value())
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())

	client = mlflow.tracking.MlflowClient()
	mlflow.set_experiment('Semantic-RAG-Search') 
	mlflow.set_tracking_uri(settings.MLFLOW_HOST.get_secret_value())
	mlflow.llama_index.autolog()

	# uvicorn.run(application, host = "0.0.0.0", port = 8000) MLFLOW