import pandas



def ontologyToDataframe(ontology) -> pandas.DataFrame:

	data = []

	for chunk in ontology:

		for entity in chunk['ontology'].ontology:

			record = {
				'concept_1': entity.concept_1,
				'concept_2': entity.concept_2,
				'relationship': entity.relationship,
				'chunk_id': chunk['metadata']['id'],
				'chunk_source': chunk['metadata']['file_name']
			}

			data += [record]

	dataframe = pandas.DataFrame(data)

	return dataframe