import json
import os


class ConfigLoader(object):
    def __init__(self, config_path='./config/', config_name='config.json'):
        self.config_path = str(config_path)
        self.config_name = str(config_name)
        self.config = None
        self.load_config()

    def load_config(self):

        config_path = self.config_path + self.config_name
        if os.path.exists(config_path) is False:
            print("The configuration you provided does not exists. ({0})".format(config_path))
            return

        with open(config_path, 'r') as json_file:
            self.config = json.loads(json_file.read())
        print("Config loaded successfully.")
