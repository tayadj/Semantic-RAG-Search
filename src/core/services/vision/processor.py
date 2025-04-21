import llama_index
import llama_index.llms.openai



class VisionProcessor:

	def __init__(self, model):

		self.model = model

		self.prompt = llama_index.core.prompts.PromptTemplate(
			"You are an image alternative text generator. "
			"Your task is to generate a clear, concise, and accessible alternative text description of the image. \n"
			"Thought 1: Focus on identifying the primary subject(s) in the image, such as people, objects, landmarks, or natural elements. \n"
			"\tConsider what is most important for conveying the essence of the image.\n\n"
			"Thought 2: Describe the setting and context, including background details, colors, lighting, and composition. \n"
			"\tHighlight any notable visual cues that give insight into the mood or environment of the scene.\n\n"
			"Thought 3: Ensure the alternative text is informative for users with visual impairments by providing a coherent and succinct summary of the visual content. \n\n"
			"Format your output as a single, plain text description that captures the key visual details and overall context of the image.\n\n"
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
