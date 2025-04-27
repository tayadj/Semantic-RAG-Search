



def serverSetup(application, system):

	@application.head('/ontology')
	async def ontology_endpoint():

		await system.ontology_pipeline()