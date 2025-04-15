import networkx
import pandas
import pyvis



def ontologyVisualization(settings, ontology_dataframe):

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