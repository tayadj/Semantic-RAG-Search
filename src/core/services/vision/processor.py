

class VisionProcessor:

	# Implement processing image content to textual content

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
