import time
from Utils import Cluster_Methods as cm
from Utils import Create_Graph as cg
from Utils import Extract_Features as ef
import glob
import numpy as np
import concurrent.futures
import pandas as pd
import json
import time



configs_path = "user/configs/Configs.json"

with open(configs_path, 'r') as j:
    json_data = json.load(j)
    shape_predictor_path = json_data["shape_predictor_path"]
    files_path = json_data["dataset_path"]
    files_exit_path = json_data["dataset_exit_encodings_path"]
    process_qtd = json_data["number_of_process"]
    back_up_percentage = json_data["back_up_percentage"]

initial_time = time.time()

images_path = glob.glob(f"{files_path}/*")
files_path_lists = np.array_split(images_path, process_qtd)
result_df = pd.DataFrame()

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(ef.extract_face_features, files_path_lists, [files_exit_path] * process_qtd,
                           range(process_qtd), [shape_predictor_path] * process_qtd, [back_up_percentage] * process_qtd)

    for result in results:
        result_df = pd.concat([result_df, result])

result_df.reset_index(drop=True, inplace=True)
result_df.index += 1

exit_var = time.time()
print(f"extravtion: {exit_var - initial_time}")

initial_time = time.time()


faces_data_graph = cg.generate_conections(result_df[["encoding"]], 0.5, True)
exit_var = time.time()
print(f"cobection: {exit_var - initial_time}")

initial_time = time.time()

G = cg.create_graph(faces_data_graph, plot_graph=False)
exit_var = time.time()
print(f"Graph: {exit_var - initial_time}")
initial_time = time.time()

G, clustered_df = cm.cluster_cw(G, cg.get_graph_edges_value(G))
exit_var = time.time()

print(f"cluster: {exit_var - initial_time}")

clustered_df.to_csv(f"{'/'.join(files_exit_path.split('/')[:-1])}/result.csv")
# for i in G.nodes():
#     print(i, G.nodes()[i]["labels"])
