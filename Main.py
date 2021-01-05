import time
from Utils import Cluster_Methods as cm
from Utils import Create_Graph as cg
from Utils import Extract_Features as ef
import glob
import numpy as np
import concurrent.futures
import pandas as pd
import json

configs_path = "/opt/project/configurations/Configs.json"

with open(configs_path, 'r') as j:
    json_data = json.load(j)
    shape_predictor_path = json_data["shape_predictor_path"]
    files_path = json_data["dataset_path"]
    files_exit_path = json_data["dataset_exit_encodings_path"]
    process_qtd = json_data["number_of_process"]
    back_up_percentage = json_data["back_up_percentage"]

images_path = glob.glob(f"{files_path}/*")

files_path_lists = np.array_split(images_path, process_qtd)
print(files_path_lists)
result_df = pd.DataFrame()

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(ef.extract_face_features, files_path_lists, [files_exit_path] * process_qtd,
                           range(process_qtd), [shape_predictor_path] * process_qtd, [back_up_percentage] * process_qtd)

    for result in results:
        result_df = pd.concat([result_df, result])

result_df.reset_index(drop=True, inplace=True)
result_df.index += 1
# while images_path:
#     threads_list = []
#
#     initial_lists = np.array_split(images_path[:process_qtd * percentage_qtd], process_qtd)
#     files = files[process_qtd * percentage_qtd:]
#     a = 0
#     for i in range(process_qtd):
#         t = multiprocessing.Process(target=ef.extract_face_features,
#                                     args=(images_path, files_path, 't' + str(i + last_process)))
#         t.start()
#         threads_list.append(t)
#         a = last_process + i
#     last_process = a
#
#     del initial_lists
#
#     for t in threads_list:
#         t.join()

# df = ef.extract_face_features(images_path, files_path, "t0")
faces_data_graph = cg.generate_conections(result_df[["encoding"]], 0.5, True)
G = cg.create_graph(faces_data_graph, plot_graph=False)
G, clustered_df = cm.cluster_cw(G, cg.get_graph_edges_value(G))

for i in G.nodes():
    print(i, G.nodes()[i]["labels"])
