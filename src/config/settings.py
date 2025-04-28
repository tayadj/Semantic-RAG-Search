import base64
import json
import os
import pydantic
import pydantic_settings



class Settings(pydantic_settings.BaseSettings):
	
	model_config = pydantic_settings.SettingsConfigDict(
		env_file = os.path.dirname(__file__) + '/.env',
		env_file_encoding = 'utf-8',
		extra = 'ignore'
	)

	OPENAI_API_KEY: pydantic.SecretStr

	LOCAL_STORAGE_URL: pydantic.SecretStr

	MLFLOW_HOST: pydantic.SecretStr
	MLFLOW_EXPERIMENT: pydantic.SecretStr
	MLFLOW_MODEL: pydantic.SecretStr

	SERVER_HOST: pydantic.SecretStr
	SERVER_PORT: pydantic.SecretStr