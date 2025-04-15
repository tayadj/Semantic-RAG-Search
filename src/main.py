import networkx
import pandas
import pyvis

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

	ontology_dataframe = pandas.read_csv(settings.LOCAL_STORAGE_URL.get_secret_value() + '_ontology.csv')
	
	ontology_graph = networkx.Graph()
	ontology_nodes = pandas.concat([ontology_dataframe['concept_1'], ontology_dataframe['concept_2']], axis = 0).unique()
	
	for node in ontology_nodes:

		ontology_graph.add_node(
			str(node)
		)

	for index, row in ontology_dataframe.iterrows():

		ontology_graph.add_edge(
			str(row['concept_1']),
			str(row['concept_2']),
			title = str(row['relationship'])
		)

	print(ontology_graph)

	network = pyvis.network.Network(
		notebook = False,
		cdn_resources = 'remote',
		height = '900px',
		width = '100%',
		select_menu = True,
		filter_menu = False
	)
	network.from_nx(ontology_graph)
	network.force_atlas_2based(central_gravity = 0.015, gravity = -31)
	network.show_buttons(filter_ = ["physics"])
	network.show(settings.LOCAL_STORAGE_URL.get_secret_value() + '_visualisation.html', notebook = False)