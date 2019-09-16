import yaml
import os

class Parser():
    def __init__(self):
        self.file_name = os.getcwd()
        self.file_name = os.path.join(self.file_name, 'config/vision_config.yaml')
        self.data = None
        with open(self.file_name) as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)

    def write_values(self, word, index, val):
        array = self.data['object'][0]['ball'][word]
        array[index] = val

        self.data['object'][0]['ball'][word] = array

        with open(self.file_name, 'w') as f:
            data = yaml.dump(self.data, f)
        
