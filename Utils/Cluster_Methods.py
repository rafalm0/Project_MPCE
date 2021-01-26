import dlib
import pandas as pd


def simple_cluster(df: pd.DataFrame):
    encodings = [dlib.vector(i) for i in df["encoding"].values.tolist()]
    labels = dlib.chinese_whispers_clustering(encodings, 0.5)
    df["cluster"] = labels
    return df


def cluster_cw(G, edge_labels):
    chinise_whipers_list = [(k[0], k[1], edge_labels[k]) for k in edge_labels]
    clusters = dlib.chinese_whispers(chinise_whipers_list)

    print(len(G))
    results_df = pd.DataFrame()
    results_df['ID'] = range(len(G) + 1)
    results_df['labels'] = clusters
    results_df.set_index('ID', inplace=True)
    results_df = results_df.reindex(G.nodes())
    results_df['labels'] = pd.Categorical(results_df['labels'])

    for i in G.nodes():
        G.nodes()[i]["labels"] = results_df.loc[i]["labels"]

    return G, results_df
