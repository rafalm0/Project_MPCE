import time
from Utils import Cluster_Methods as cm
from Utils import Create_Graph as cg
from Utils import Extract_Features as ef

path = "/opt/project/dataset/train"
df = ef.extract_face_features(path)
faces_data_graph = cg.generate_conections(df[["encoding"]], 0.5, True)
G = cg.create_graph(faces_data_graph, plot_graph=False)
G, clustered_df = cm.cluster_cw(G, cg.get_graph_edges_value(G))

for i in G.nodes():
    print(i, G.nodes()[i]["labels"])