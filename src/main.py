import fastapi
import llama_index
import mlflow
import pandas
import pydantic
import time
import uvicorn

import config
import core
import data
import utils



application = fastapi.FastAPI()



@application.head('/ontology')
async def ontology_pipeline():

	documents = database.local_connector.load()

	image_documents = [document for document in documents if isinstance(document, llama_index.core.schema.ImageDocument)]
	described_image_documents = await engine.generate_description(image_documents)

	documents = [document for document in documents if not isinstance(document, llama_index.core.schema.ImageDocument)]
	documents = documents + described_image_documents

	documents_dataframe = utils.converters.documentsToDataframe(documents)

	mlflow.log_artifact(
		settings.LOCAL_STORAGE_URL.get_secret_value() + '_documents.csv',
		artifact_path = 'documents'
	)

	ontology = await engine.generate_ontology(documents_dataframe)
	ontology_dataframe = utils.converters.ontologyToDataframe(ontology)
	ontology_dataframe.to_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv', index = False)

	utils.visualization.ontologyVisualization(settings, ontology_dataframe)



@application.head('/knowledge')
async def knowledge_pipeline():

	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')

	knowledge_index = await engine.generate_knowledge(documents_dataframe, ontology_dataframe)

	model = mlflow.llama_index.log_model(
		knowledge_index,
		artifact_path = 'llama_index',
		engine_type = 'query',
		registered_model_name = settings.MLFLOW_MODEL.get_secret_value()
	)



class InferenceRequest(pydantic.BaseModel):

	query: str

@application.post('/inference')
async def inference_pipeline(request: InferenceRequest):

	global inference_step

	start_time = time.perf_counter()

	model_name = settings.MLFLOW_MODEL.get_secret_value()
	model_version = client.get_registered_model(model_name).latest_versions[0].version
	model_uri = f'models:/{model_name}/{model_version}'

	knowledge_index = mlflow.llama_index.load_model(model_uri)

	query_engine = knowledge_index.as_query_engine(include_text = True, response_mode = 'tree_summarize')
	response = await query_engine.aquery(request.query)

	end_time = time.perf_counter()

	latency = end_time - start_time

	mlflow.log_metric("latency", latency, step = inference_step)
	inference_step += 1 # race condition

	return {'response': str(response)}



@application.head('/evaluation')
async def evaluation_pipeline():

	model_name = settings.MLFLOW_MODEL.get_secret_value()
	model_version = client.get_registered_model(model_name).latest_versions[0].version
	model_uri = f'models:/{model_name}/{model_version}'

	evaluation_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_evaluation.csv')

	results = mlflow.evaluate(
		model = model_uri,
		data = evaluation_dataframe,
		targets = 'ground_truth',
		model_type = 'question-answering',
		extra_metrics = [ 
			mlflow.metrics.latency()
		]
	)



if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database(local_storage_url = settings.LOCAL_STORAGE_URL.get_secret_value())
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())

	mlflow.set_experiment('Semantic-RAG-Search') 
	mlflow.set_tracking_uri(settings.MLFLOW_HOST.get_secret_value())
	mlflow.llama_index.autolog()

	client = mlflow.tracking.MlflowClient()
	run = mlflow.start_run()
	inference_step = 0

	try:

		uvicorn.run(application, host = "0.0.0.0", port = 8000)

	finally:

		mlflow.end_run()