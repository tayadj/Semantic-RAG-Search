import llama_index.core



class LocalConnector:

	def __init__(self, URL: str):

		self.URL = URL
		
	def load(self):

		reader = llama_index.core.SimpleDirectoryReader(input_dir = self.URL)
		documents = reader.load_data()

		return documents