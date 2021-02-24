import json
import numpy as np


class ConfigJsonValues:
    configs_path = "user/configs/Configs.json"
    with open(configs_path, 'r') as j:
        json_data = json.load(j)
    shape_predictor_path = json_data["shape_predictor_path"]
    dataset_output_path = json_data["dataset_output_path"]
    back_up_percentage = json_data["back_up_percentage"]
    process_qtd = json_data["number_of_process"]
    files_path = json_data["dataset_path"]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
