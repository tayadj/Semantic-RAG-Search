import llama_index
import llama_index.llms.openai



class SonitionProcessor:

	def __init__(self, model):

		self.model = model

		self.prompt = llama_index.core.prompts.PromptTemplate(
			"You are an audio transcription engine.\n"
			"Your task is to accurately convert the provided audio file into a clear, coherent, and well-formatted text transcription.\n"
			"Thought 1:"
			"\tListen attentively to the audio content and identify all spoken words and phrases.\n"
			"\tFocus on distinguishing between different speakers and capturing all dialogue and narration accurately.\n"
			"Thought 2:"
			"\tConsider the context and pace of the speech.\n"
			"\tPay attention to pauses, intonation, and any background sounds that provide additional context or emotional nuance.\n"
			"Thought 3:"
			"\tEnsure the transcription is accessible and easy to read by applying proper punctuation, capitalization, and formatting.\n"
			"\n"
			"Format your output as a single, plain text transcription that faithfully represents the original audio content, preserving both its structure and details.\n"
			"\n"
			"Output: "
		)


