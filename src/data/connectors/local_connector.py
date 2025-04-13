from llama_index.core import SimpleDirectoryReader



class LocalConnector:

	def __init__(self, URL: str):

		self.URL = URL
		
	def load(self):

		reader = SimpleDirectoryReader(input_dir = self.URL)
		documents = reader.load_data()

		print(documents)