import asyncio
import fastapi
import llama_index
import mlflow
import pandas
import time
import uvicorn

import config
import core
import data
import utils



class System:

	def __init__(self, settings, database, engine, tracker):

		self.settings = settings
		self.database = database
		self.engine = engine
		self.tracker = tracker

		self.model_name = settings.MLFLOW_MODEL.get_secret_value()
		self.model_version = None # self.tracker.get_registered_model(self.model_name).latest_versions[0].version
		self.model_uri = None  # f'models:/{self.model_name}/{self.model_version}'
		self.model = None # mlflow.llama_index.load_model(self.model_uri)

		self.time = 0
		self.time_lock = asyncio.Lock()

	async def ontology_pipeline(self):

		documents = self.database.local_connector.load()

		image_documents = [document for document in documents if isinstance(document, llama_index.core.schema.ImageDocument)]
		described_image_documents = await self.engine.generate_description(image_documents)

		documents = [document for document in documents if not isinstance(document, llama_index.core.schema.ImageDocument)]
		documents = documents + described_image_documents

		documents_dataframe = utils.converters.documentsToDataframe(documents)
		documents_dataframe.to_csv(self.settings.LOCAL_STORAGE_URL.get_secret_value() + '/.meta/documents.csv', index = False)

		mlflow.log_artifact(
			self.settings.LOCAL_STORAGE_URL.get_secret_value() + '/.meta/documents.csv',
			artifact_path = 'documents'
		)

		ontology = await self.engine.generate_ontology(documents_dataframe)
		ontology_dataframe = utils.converters.ontologyToDataframe(ontology)
		ontology_dataframe.to_csv(self.settings.LOCAL_STORAGE_URL.get_secret_value() + '/.meta/ontology.csv', index = False)

		utils.visualization.ontologyVisualization(self.settings, ontology_dataframe)

	async def knowledge_pipeline(self):

		documents = self.database.local_connector.load()
		documents_dataframe = utils.converters.documentsToDataframe(documents)

		ontology_dataframe = pandas.read_csv(self.settings.LOCAL_STORAGE_URL.get_secret_value() + '/.meta/ontology.csv')

		self.model = await self.engine.generate_knowledge(documents_dataframe, ontology_dataframe)

		model = mlflow.llama_index.log_model(
			self.model,
			artifact_path = 'llama_index',
			engine_type = 'query',
			registered_model_name = self.settings.MLFLOW_MODEL.get_secret_value()
		)

	async def inference_pipeline(self, request: utils.models.InferenceRequest):

		start_time = time.perf_counter()

		if self.model is None:

			self.model_version = self.tracker.get_registered_model(self.model_name).latest_versions[0].version
			self.model_uri = f'models:/{self.model_name}/{self.model_version}'
			self.model = mlflow.llama_index.load_model(self.model_uri)

		query_engine = self.model.as_query_engine(include_text = True, response_mode = 'tree_summarize')
		response = await query_engine.aquery(request.query)

		end_time = time.perf_counter()

		latency = end_time - start_time

		async with self.time_lock:
			
			mlflow.log_metric("latency", latency, step = self.time)
			self.time += 1

		return {'response': str(response)}

	async def evaluation_pipeline(self):

		if self.model is None:

			self.model_version = self.tracker.get_registered_model(self.model_name).latest_versions[0].version
			self.model_uri = f'models:/{self.model_name}/{self.model_version}'
			self.model = mlflow.llama_index.load_model(self.model_uri)

		evaluation_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '/.meta/evaluation.csv')

		results = mlflow.evaluate(
			model = self.model_uri,
			data = evaluation_dataframe,
			targets = 'ground_truth',
			extra_metrics = [ 
				mlflow.metrics.latency(),
				mlflow.metrics.token_count(),
				mlflow.metrics.genai.answer_correctness("openai:/gpt-4o-mini")
			],
			evaluator_config = {
				'col_mapping': {
					'inputs': 'query_str'
				}
			}
		)



if __name__ == '__main__':

	settings = config.Settings()
	database = data.Database(local_storage_url = settings.LOCAL_STORAGE_URL.get_secret_value())
	engine = core.Engine(settings.OPENAI_API_KEY.get_secret_value())
	tracker = mlflow.tracking.MlflowClient()

	system = System(settings, database, engine, tracker)
	application = fastapi.FastAPI()
	utils.setup.serverSetup(application, system)

	try:

		mlflow.set_experiment('Semantic-RAG-Search') 
		mlflow.set_tracking_uri(settings.MLFLOW_HOST.get_secret_value())
		mlflow.llama_index.autolog()
		mlflow.start_run()
		uvicorn.run(application, host = "0.0.0.0", port = 8000)

	finally:

		mlflow.end_run()