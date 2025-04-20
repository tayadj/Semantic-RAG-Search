from .connectors import LocalConnector


class Database:

	def __init__(self, config: dict):

		self.local_connector = LocalConnector(config.get('LOCAL_STORAGE_URL'))