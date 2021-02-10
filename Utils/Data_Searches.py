import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2
import os
import re


def show_cluster_random_faces(df_l: pd.DataFrame, cluster_number: int, faces_count: int = 16,
                              figure_size: tuple = (16, 16), img_size: tuple = (96, 96)):
    idxs = df_l[df_l["cluster"] == cluster_number]
    idx = idxs.sample(min(faces_count, len(idxs)))
    fig = plt.figure(figsize=figure_size)

    size = math.sqrt(faces_count)
    if size % 1 > 0:
        size = int(size) + 1
    size = int(size)

    for i in range(len(idx)):
        line = idx.iloc[i]
        image = cv2.imread(line["imagePath"])
        top, right, bottom, left = line["face_locations"]
        face = image[int(top):int(bottom), int(left):int(right)]
        face = cv2.resize(face, img_size)
        plt.subplot(size, size, i + 1)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        plt.axis("off")
        plt.imshow(face)
        plt.title(line["imagePath"].split("/")[-1][:-4])


def show_clusters_main_face(nome_do_caso: str, tamanho_da_imagem: tuple = (8, 14), cols: int = 6,
                            top_distance: float = 10):
    path = f"user/dataset/exit_data/{nome_do_caso}/cluster_imgs"
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    px = 1 / plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(tamanho_da_imagem[0] * px, tamanho_da_imagem[1] * px))
    plt.subplots_adjust(top=top_distance)

    rows = len(files) - cols

    for i, file in enumerate(files):
        plt.subplot(rows, cols, i + 1)
        img = cv2.imread(f"{path}/{file}")
        # cv2_imshow(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis("off")
        plt.imshow(img)
        plt.title(file[:-4])
    plt.show()


def show_cluster_connections(cluster: int, nome_do_caso: str, conection_df: pd.DataFrame,
                             tamanho_da_imagem: tuple = (8, 14), img_size: tuple = (96, 96), cols: int = 6,
                             top_distance: float = 10):
    path = f"user/dataset/exit_data/{nome_do_caso}/cluster_imgs"
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    initial_file = files[cluster]

    fig = plt.figure(figsize=(2, 2))
    plt.subplot(1, 1, 1)
    img = cv2.imread(f"{path}/{initial_file}")
    # cv2_imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(img)
    plt.title(f"cluster: {initial_file[:-4]}")
    plt.show()

    px = 1 / plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(tamanho_da_imagem[0] * px, tamanho_da_imagem[1] * px))
    plt.subplots_adjust(top=top_distance)

    rows = len(files) - cols

    for i, file in enumerate(files):
        if (file == initial_file):
            continue

        plt.subplot(rows, cols, i + 1)
        img = cv2.imread(f"{path}/{file}")
        img = cv2.resize(img, img_size)
        # cv2_imshow(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis("off")
        plt.imshow(img)

        line = conection_df[((conection_df["cluster_x"] == cluster) & (conection_df["cluster_y"] == i)) |

                            ((conection_df["cluster_x"] == i) & (conection_df["cluster_y"] == cluster))]

        plt.title(f"cluster: {file[:-4]}\nconexão: {0 if len(line) == 0 else line.iloc[0]['occurrence']}")
    plt.show()
