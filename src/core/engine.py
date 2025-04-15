from .services import OntologyProcessor


class Engine:

	def __init__(self, model):

		self.model = model
		self.ontology_processor = OntologyProcessor(model)

