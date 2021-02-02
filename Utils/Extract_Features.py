import face_recognition
import cv2
from imutils.face_utils import FaceAligner
import dlib
import pandas as pd
import pickle
import os
import numpy as np
# from tqdm import tqdm

def save_result_json(df: pd.DataFrame, result_json: dict):
    clusters = np.sort(df["cluster"].unique())

    graph_df = df.drop(columns=["face_locations", "encoding"])
    graph_df = graph_df.merge(graph_df, on="imagePath")
    graph_df[["cluster_x", "cluster_y"]] = graph_df[["cluster_x", "cluster_y"]].apply(pd.to_numeric)

    graph_df = graph_df.query("cluster_x < cluster_y")

    graph_df.drop_duplicates(inplace=True)

    occurrences = graph_df.groupby(by="cluster_x").apply(lambda a: (a["cluster_y"].tolist()))
    graph_df = graph_df.sort_values(by=['cluster_x', 'cluster_y']).drop_duplicates(
        subset=['cluster_x', 'cluster_y']).reset_index(drop=True)

    result = []
    for i, occurrence in enumerate(occurrences):
        fitered = np.unique(np.array(occurrence), return_counts=True)
        result.extend(fitered[1])

    graph_df["occurrence"] = result

    self_pointg_df = {"imagePath": [], "cluster_x": [], "cluster_y": [], "occurrence": []}
    for cluster in df["cluster"].unique():
        im_path = df[df["cluster"] == cluster]["imagePath"].values[0]
        self_pointg_df["imagePath"].append(im_path)
        self_pointg_df["cluster_x"].append(cluster)
        self_pointg_df["cluster_y"].append(cluster)
        self_pointg_df["occurrence"].append(1)

    aux_df = pd.DataFrame(self_pointg_df)
    graph_df = pd.concat([graph_df, aux_df])

    all_clusters_list = list(range(len(clusters)))
    for i in clusters:
        group = graph_df.groupby(by="cluster_x").get_group(i)
        images = group["imagePath"]
        # values = group.set_index("cluster_x").to_cdict("list")
        dict_results = \
            graph_df.groupby(by="cluster_x").get_group(i)[["cluster_y", "occurrence"]].set_index("cluster_y").to_dict()[
                "occurrence"]

        for j in np.setdiff1d(np.array(all_clusters_list), np.array([*dict_results])):
            dict_results[j] = 0

        result_json[i] = {"values": {"image_list": images.to_list(), "clusters_info": dict_results}}

    return result_json


def generate_cluster_faces(df: pd.DataFrame, images_exit_path: str) -> dict:
    clusters = np.sort(df["cluster"].unique())
    cluster_groups = df.groupby("cluster")

    result_json = {}
    images_exit_path = f"{images_exit_path}/cluster_imgs"
    if not os.path.exists(images_exit_path):
        os.mkdir(images_exit_path)

    for i in clusters:
        group_encodings = cluster_groups.get_group(i)["encoding"]
        distances = face_recognition.face_distance(group_encodings.to_list(), group_encodings.mean())
        index = np.argmin(distances, axis=0)
        line = cluster_groups.get_group(i).iloc[index]
        result_json[i] = {"main_image": line["imagePath"], "main_image_loc": line["face_locations"]}
        image = cv2.imread(line["imagePath"])
        (top, right, bottom, left) = line["face_locations"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (128, 128))
        cv2.imwrite(f"{images_exit_path}/{i}.png", face)

    return result_json


def path_creation_verification(path: str):
    if os.path.exists(path):
        return

    path_splited = path.split("/") if "/" in path else path.split("\\")

    full_path = ""
    for directory in path_splited:
        full_path = f"{full_path}/{directory}"
        if not os.path.exists(full_path):
            os.mkdir(full_path)


def extract_face_features(images_path: list, images_exit_path: str, process_number: str,
                          shape_predictor_path: str, back_up_percentage: float,
                          normalize: bool = True, print_key: bool = True, save_pickle: bool = True,
                          save_csv: bool = True,
                          detection_method: str = "cnn"
                          ) -> pd.DataFrame:
    print(f"Process Number {process_number} was started")

    predictor = dlib.shape_predictor(shape_predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    images_exit_path = f"{images_exit_path}/encodings"
    if not os.path.exists(images_exit_path):
        os.mkdir(images_exit_path)

    to_back_up_qtd = int(len(images_path) + 1)

    data = {"imagePath": [], "face_locations": [], "encoding": []}
    for (i, imagePath) in enumerate(images_path):
        if print_key:
            print(f"[INFO] processing {imagePath} , {i + 1}/{len(images_path)}")
            print(imagePath)

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = face_recognition.face_locations(rgb, model=detection_method)
        if len(boxes) == 0:
            continue

        if normalize:
            encodings = []

            for i, box in enumerate(boxes):
                top, right, bottom, left = box
                rec = dlib.rectangle(top=top, right=right, bottom=bottom, left=left)
                faceAligned = fa.align(image, gray, rec)
                faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                # faceAligned = imutils.resize(faceAligned, width= right - left, height= bottom - top)
                # if plot_faces:
                #   cv2_imshow(faceAligned)
                # cv2_imshow(a)

                encodings.extend(face_recognition.face_encodings(faceAligned, [(0, 256, 256, 0)]))
        else:
            encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, enc) in zip(boxes, encodings):
            data["imagePath"].append(imagePath)
            data["face_locations"].append(box)
            data["encoding"].append(enc)

        if back_up_percentage > 0 and i % to_back_up_qtd == 0:

            if save_csv:
                aux_df = pd.DataFrame(data)
                aux_df.to_csv(f"{images_exit_path}/image_encondings_{process_number}.csv")
                del aux_df
            if save_pickle:
                save_pickle_at(data, images_exit_path, process_number)

    data = pd.DataFrame(data)
    data.index += 1

    while not os.path.exists(images_exit_path):
        path_creation_verification(images_exit_path)

    if save_csv:
        data.to_csv(f"{images_exit_path}/image_encondings_{process_number}.csv")
    if save_pickle:
        save_pickle_at(data, images_exit_path, process_number)

    print(f"Process Number {process_number} has ended")

    return data


def save_pickle_at(data, path, process_text):
    f = open(f"{path}/image_encondings_{process_text}.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def load_pickle(path, process_text) -> pd.DataFrame:
    return pickle.loads(open(f"{path}/image_encondings_{process_text}.pickle", "rb").read())

    # f.write(pickle.dumps(G))
    # f.close()


if __name__ == '__main__':
    # print(os.path.abspath(__file__))
    path = "/opt/project/dataset/train/bala"
    path_creation_verification(path)
    # print(os.listdir(path))
    # df = extract_face_features(path)
    # print(df)
