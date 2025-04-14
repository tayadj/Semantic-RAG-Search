import config
import core
import data
import utils



if __name__ == '__main__':

	settings = config.Settings()

	database = data.Database({'LOCAL_STORAGE_URL': settings.LOCAL_STORAGE_URL.get_secret_value()})
	documents = database.local_connector.load()

	dataframe = utils.converters.documentsToDataframe(documents)

	print(dataframe)