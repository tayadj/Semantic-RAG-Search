import llama_index
import pydantic
import typing



class OntologyProcessor:

	class OntologyEntity(pydantic.BaseModel):

		concept_1: str
		concept_2: str
		relationship: str

	class Ontology(pydantic.BaseModel):

		ontology: typing.List["OntologyEntity"]

	def __init__(self, model):

		self.model = model

		self.prompt = llama_index.core.prompts.PromptTemplate(
			"You are a network graph maker who extracts terms and their relations from a given context. "
			"You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
			"of terms mentioned in the given context. These terms should represent the key concepts as per the context.\n"
			"Thought 1:"
			"\tWhile traversing through each sentence, Think about the key terms mentioned in it.\n"
			"\tTerms may include object, entity, location, organization, person, condition, acronym, documents, service, concept, etc.\n"
			"\tTerms should be as atomistic as possible\n"
			"Thought 2:"
			"\tThink about how these terms can have one on one relation with other terms.\n"
			"\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
			"\tTerms can be related to many other terms\n"
			"Thought 3:"
			"\tFind out the relation between each such related pair of terms.\n"
			"\n"
			"Format your output as a list of json. Each element of the list contains a pair of terms"
			"and the relation between them, like the follwing: \n"
			"[\n"
			"   {\n"
			"       \"concept_1\": \"A concept from extracted ontology\",\n"
			"       \"concept_2\": \"A related concept from extracted ontology\",\n"
			"       \"relationship\": \"relationship between the two concepts, concept_1 and concept_2 in one or two sentences\"\n"
			"   }, {...}\n"
			"]\n"
			"\n"
			"Context: ```{context}```\n"
			"\n"
			"Output: "
		)

	async def process(self, text):

		result = await (
			self.model
				.as_structured_llm(self.Ontology)
				.acomplete(self.prompt.format(context = text))
				.raw			
		)

		return result
