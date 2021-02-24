from Utils import Cluster_Methods as cm
from Utils import Create_Graph as cg
from Utils import Extract_Features as ef
from Utils.UtilMethods import NpEncoder
from os import walk

import glob
import numpy as np
import concurrent.futures
import pandas as pd
import json
import time
import os
from Utils.UtilMethods import ConfigJsonValues


def comecar_processamento(nome_do_caso) -> pd.DataFrame:
    generate_graph = False

    dataset_output_path = f"{ConfigJsonValues.dataset_output_path}"
    if not os.path.exists(dataset_output_path):
        os.mkdir(dataset_output_path)

    dataset_output_path = f"{dataset_output_path}/{nome_do_caso}"
    if not os.path.exists(dataset_output_path):
        os.mkdir(dataset_output_path)

    if os.path.exists(f"{dataset_output_path}/image_encondings_{nome_do_caso}.pickle"):
        return ef.load_pickle(f"{dataset_output_path}/image_encondings_{nome_do_caso}.pickle")

    initial_time = time.time()
    files_path = f"{ConfigJsonValues.files_path}/{nome_do_caso}"
    print(files_path)
    if not os.path.exists(files_path):
        print("Arquivos de entrada dos casos não existem")
        return None

    import glob
    images_path = []
    image_types = ["png", "jpg", "jpeg"]
    for ext in image_types:
        images_path.extend(list(glob.iglob(f'{files_path}/**/*.{ext}', recursive=True)))

    files_path_lists = np.array_split(images_path, ConfigJsonValues.process_qtd)
    result_df = pd.DataFrame()

    images_exit_path = f"{dataset_output_path}/encodings"
    if not os.path.exists(images_exit_path):
        os.mkdir(images_exit_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(ef.extract_face_features, files_path_lists,
                               [dataset_output_path] * ConfigJsonValues.process_qtd,
                               range(ConfigJsonValues.process_qtd))

        for result in results:
            result_df = pd.concat([result_df, result])

    result_df.reset_index(drop=True, inplace=True)
    result_df.index += 1

    exit_var = time.time()
    print(f"extraction: {exit_var - initial_time}")

    initial_time = time.time()

    if (generate_graph):
        faces_data_graph = cg.generate_conections(result_df[["encoding"]], 0.5, True)
        exit_var = time.time()
        print(f"conection: {exit_var - initial_time}")

        initial_time = time.time()

        G = cg.create_graph(faces_data_graph, plot_graph=False)
        exit_var = time.time()
        print(f"Graph: {exit_var - initial_time}")
        initial_time = time.time()

        G, clustered_df = cm.cluster_cw(G, cg.get_graph_edges_value(G))
        exit_var = time.time()

        print(f"cluster: {exit_var - initial_time}")

        result_df.index.name = "id"
        result_df["cluster"] = clustered_df["labels"]
        del clustered_df
    else:
        result_df = cm.simple_cluster(result_df)

    result_df.to_csv(f"{dataset_output_path}/result.csv")
    ef.save_pickle_at(result_df, f"{dataset_output_path}/{nome_do_caso}.pickle")
    ef.generate_cluster_faces(result_df, dataset_output_path)
    return result_df


def get_casos() -> list:
    return os.listdir(ConfigJsonValues.dataset_output_path)


def generate_cluster_connections(df: pd.DataFrame):
    connection_df = df.drop(columns=["face_locations", "encoding"])
    connection_df = connection_df.merge(connection_df, on="imagePath")
    connection_df = connection_df.query("cluster_x < cluster_y")
    connection_df = connection_df.copy()
    connection_df.drop_duplicates(inplace=True)
    occurrences = connection_df.groupby(by="cluster_x").apply(lambda a: (a["cluster_y"].tolist()))
    connection_df = connection_df.sort_values(by=['cluster_x', 'cluster_y']).drop_duplicates(
        subset=['cluster_x', 'cluster_y']).reset_index(drop=True)
    result = []
    for i, occurrence in enumerate(occurrences):
        fitered = np.unique(np.array(occurrence), return_counts=True)
        result.extend(fitered[1])

    connection_df["occurrence"] = result

    self_pointg_df = {"imagePath": [], "cluster_x": [], "cluster_y": [], "occurrence": []}
    for cluster in df["cluster"].unique():
        im_path = df[df["cluster"] == cluster]["imagePath"].values[0]
        self_pointg_df["imagePath"].append(im_path)
        self_pointg_df["cluster_x"].append(cluster)
        self_pointg_df["cluster_y"].append(cluster)
        self_pointg_df["occurrence"].append(1)

    aux_df = pd.DataFrame(self_pointg_df)
    connection_df = pd.concat([connection_df, aux_df])
    return connection_df


if __name__ == '__main__':
    # print("sadsad")
    comecar_processamento("friends")
# result_json = ef.generate_cluster_faces(result_df, files_exit_path)
# result_json = ef.save_result_json(result_df, result_json)

# json_object = json.dumps(result_json, indent=4, cls=NpEncoder)
# del result_json

# with open(f"{files_exit_path}/result.json", "w") as outfile:
#     outfile.write(json_object)

# del result_df


# for i in G.nodes():
#     print(i, G.nodes()[i]["labels"])
