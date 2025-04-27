import utils



def serverSetup(application, system):

	@application.head('/ontology')
	async def ontology_endpoint():

		await system.ontology_pipeline()
	
	@application.head('/knowledge')
	async def knowledge_endpoint():

		await system.knowledge_pipeline()

	@application.post('/inference')
	async def inference_endpoint(request: utils.models.InferenceRequest):

		return await system.inference_pipeline(request)

	@application.head('/evaluation')
	async def evaluation_endpoint():

		await system.evaluation_pipeline()