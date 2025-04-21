from .connectors import LocalConnector


class Database:

	def __init__(self, **config: any):

		if 'local_storage_url' in config: 

			self.local_connector = LocalConnector(config.pop('local_storage_url'))