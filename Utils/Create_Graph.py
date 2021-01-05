from Utils.Extract_Features import extract_face_features
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


def get_graph_edges_value(G, attribute: str = "value"):
    return nx.get_edge_attributes(G, attribute)


def generate_conections(encodings, threshold: float = 0.5, euclidean: bool = True):
    faces_data_graph = {"node_x": [], "node_y": [], "value": []}
    # threshold = 0.5
    for i, line in encodings.iterrows():
        for j, line2 in encodings.iterrows():
            # if euclidean:
            dist = np.linalg.norm(line['encoding'] - line2['encoding'])

            if threshold > dist:
                faces_data_graph["node_x"].append(i)
                faces_data_graph["node_y"].append(j)
                faces_data_graph["value"].append(dist)
            # else:
            #     dist = 1 - sm.pairwise.cosine_similarity([line['encoding']], [line2['encoding']], dense_output=True)[0][
            #         0]
            #     # dist = int(dist) if dist > 1 or dist < 0 else dist
            #
            #     faces_data_graph["node_x"].append(i)
            #     faces_data_graph["node_y"].append(j)
            #     faces_data_graph["value"].append(dist)

    return pd.DataFrame(faces_data_graph)


def plot_graph_clustering(G, clusters_data: pd.DataFrame = None, attribute: str = "value"):
    edge_label = get_graph_edges_value(G, attribute)
    pos = nx.spring_layout(G)

    if type(clusters_data) == pd.DataFrame:
        clusters_data['labels'] = pd.Categorical(clusters_data['labels'])

        nx.draw(G, with_labels=True, pos=pos, node_color=clusters_data['labels'].cat.codes, cmap=plt.cm.Set1,
                node_size=500)
    else:
        nx.draw(G, pos, with_labels=True)

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_label, font_color='red')
    plt.show()


def create_graph(graph_data, plot_graph: bool = False, source: str = "node_x", target: str = "node_y",
                 edge_attr: list = ["value"], attribute: str = "value"):
    if not type(graph_data) == pd.DataFrame:
        graph_data = pd.DataFrame(graph_data)

    G = nx.from_pandas_edgelist(graph_data, source=source, target=target, edge_attr=edge_attr)
    if plot_graph:
        plot_graph_clustering(G, attribute=attribute)

    return G


def add_new_faces(G, faces_dataframe: pd.DataFrame, new_faces_dataframe, threshold: float = 0.5, normalize: bool = True,
                  print_key: bool = False):
    if type(new_faces_dataframe) == list:
        new_faces_dataframe = extract_face_features(new_faces_dataframe, normalize=normalize, print_key=print_key)

    new_faces_dataframe.index += max(faces_dataframe.index)
    joined_data = pd.concat([faces_dataframe, new_faces_dataframe], ignore_index=False, sort=False)

    for i, line in new_faces_dataframe.iterrows():
        G.add_node(i)
        for j, line2 in faces_dataframe.iterrows():
            dist = np.linalg.norm(line['encoding'] - line2['encoding'])
            if threshold > dist:
                G.add_edge(i, j, value=dist)

    return G, joined_data

# if __name__ == '__main__':
# with open("/opt/project/dataset/image_encondings.pickle", 'rb') as file:
#     data = pickle.load(file)
# faces_data_graph = generate_conections(data[["encoding"]], 0.5, True)
# G = create_graph(faces_data_graph, plot_graph=False)
# print(clustered_df)
# for i in G.nodes():
#     print(i, G.nodes()[i]["labels"])
# G.nodes()[i]["labels"] = results_df.loc[i]["labels"]
