import config
import core
import data



if __name__ == '__main__':

	settings = config.Settings()

	connector = data.connectors.LocalConnector(settings.LOCAL_STORAGE_URL.get_secret_value())
	connector.load()