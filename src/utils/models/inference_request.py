import pydantic



class InferenceRequest(pydantic.BaseModel):

	query: str