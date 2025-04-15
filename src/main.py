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
	documents = database.local_connector.load()
	documents_dataframe = utils.converters.documentsToDataframe(documents)

	ontology = engine.generate_ontology(documents_dataframe)
	ontology_dataframe = utils.converters.ontologyToDataframe(ontology)
	ontology_dataframe.to_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv', index = False)
	'''

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')
	print(ontology_dataframe)