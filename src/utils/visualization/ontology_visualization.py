import networkx
import pandas
import pyvis
import seaborn



def ontologyVisualization(settings, ontology_dataframe):

	ontology_graph = networkx.Graph()
	ontology_nodes = pandas.concat([ontology_dataframe['concept_1'], ontology_dataframe['concept_2']], axis = 0).unique()
	
	for node in ontology_nodes:

		ontology_graph.add_node(
			str(node)
		)

	for index, record in ontology_dataframe.iterrows():

		ontology_graph.add_edge(
			str(record['concept_1']),
			str(record['concept_2']),
			title = str(record['relationship'])
		)

	communities_generator = networkx.community.girvan_newman(ontology_graph)
	next(communities_generator)
	communities = next(communities_generator)
	communities = sorted(map(sorted, communities))

	palette = seaborn.color_palette('hls', len(communities)).as_hex()
	group = 0

	for community in communities:

		color = palette.pop()
		group += 1

		for node in community:

			ontology_graph.nodes[node]['group'] = group
			ontology_graph.nodes[node]['color'] = color
			ontology_graph.nodes[node]['size'] = ontology_graph.degree[node]	

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