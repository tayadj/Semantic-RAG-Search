import llama_index
import llama_index.llms.openai



class VisionProcessor:

	def __init__(self, model):

		self.model = model

	self.prompt = llama_index.core.prompts.PromptTemplate(
		"You are an image alternative text and data extractor.\n"
		"Your task is to carefully analyze the given image and generate a clear, detailed,"
		"and accessible alternative text description that captures not only the visual content"
		"but also any embedded information.\n"
		"\n"
		"Thought 1:\n"
		"\tIdentify the primary subject(s) in the image, such as people, objects, landmarks, or natural elements.\n"
		"\tDescribe their appearance and importance.\n"
		"\n"
		"Thought 2:\n"
		"\tDescribe the setting and context in terms of background details, colors, lighting, and composition.\n"
		"\n"
		"Thought 3:\n"
		"\tSearch for and incorporate any embedded text, tabular data, diagrams, charts, or schematics that appear within the image."
		"\tEnsure that these details are clearly and coherently described, so users with visual impairments can understand all aspects of the image.\n"
		"\n"
		"Format your output as a single, plain text description that combines the visual features with any extracted textual or diagrammatic information.\n"
		"\n"
		"Output: "
	)


	def process(self, image_document):

		message = llama_index.core.llms.ChatMessage(
			role = llama_index.core.llms.MessageRole.USER,
			blocks = [
				llama_index.core.llms.TextBlock(
					text = self.prompt.format()
				),
				llama_index.core.llms.ImageBlock(
					image = image_document.image_resource.data
				)				
			]
		)

		description = self.model.chat(
			messages = [message]
        )

		described_image_text_resource = image_document.text_resource.model_copy(
			update = {
				'text': description
			}
		)
		
		described_image_document = image_document.model_copy(
			update = {
				'text_resource': described_image_text_resource
			}
		)

		return described_image_document
