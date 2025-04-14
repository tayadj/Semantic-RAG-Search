import pandas



def documentsToDataframe(documents) -> pandas.DataFrame:

	data = []

	for chunk in documents:

		record = {
			'id': chunk.id_,
			'text': chunk.text,
			**chunk.metadata
		}

		data += [record]

	dataframe = pandas.DataFrame(data)

	return dataframe