from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import networkx as nx
import random
import ast
import itertools as iter
from collections import Counter
import json

def weighted_edge_fuc():
    CSS_paper = pd.read_csv("/Users/haseebshafi/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/4. Semester/02467 - ComSocSci/Week 3/FInal_data/D8_CSS_Papers.csv")
    weighted_edge = [()]
    uniqe_combi = []
    for combinations in CSS_paper["author_ids"]:
        actual_list = ast.literal_eval(combinations)
        uniqe_combi.extend(iter.combinations(sorted(set(actual_list)), 2))

    pair_counts = Counter(uniqe_combi)

    weighted_edge = [(author1, author2, count) for (author1, author2), count in pair_counts.items()]
    
    return weighted_edge


# Build graph
G = nx.Graph()
G.add_weighted_edges_from(weighted_edge_fuc())

# attach country_code
df_final = pd.read_csv(
    "/Users/haseebshafi/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/4. Semester/02467 - ComSocSci/Week 3/Final_data/D7_CSS_Authors.csv"
)

key_col = "author_id"
df_final = df_final[df_final[key_col].isin(G.nodes())].copy()

# clean missing values
df_final["country_code"] = df_final["country_code"].replace([np.inf, -np.inf], np.nan)
df_final["country_code"] = df_final["country_code"].fillna("Unknown")

# Convert entire dataframe to dict of dicts
attr_dict = df_final.set_index(key_col).to_dict(orient="index")

# Attach all attributes
nx.set_node_attributes(G, attr_dict)

# Remove nodes with unknown country
valid_nodes = [
    n for n in G.nodes
    if G.nodes[n].get("country_code") not in [None, "Unknown"]
]

G_valid = G.subgraph(valid_nodes).copy()
largest_cc = max(nx.connected_components(G_valid), key=len)
G_valid = G_valid.subgraph(largest_cc).copy()
    

def randomize_G(G2, show_progress=False):
    G2 = G2.copy()

    def canon(a, b):
        return tuple(sorted((a, b)))

    edges = [canon(u, v) for u, v in G2.edges()]
    E = len(edges)

    it = range(E * 10)
    if show_progress:
        from tqdm.notebook import tqdm
        it = tqdm(it, desc="Edge swaps", leave=False)

    for _ in it:
        i, j = random.sample(range(E), 2)
        (u, v) = edges[i]
        (x, y) = edges[j]

        # avoid selfloops
        if u == y or x == v:
            continue

        # flip
        if random.random() < 0.5:
            u, v = v, u

        new1 = canon(u, y)
        new2 = canon(x, v)
        old1 = canon(*edges[i])
        old2 = canon(*edges[j])

        # avoid creating duplicate edges
        if new1 == new2:
            continue
        if G2.has_edge(*new1) or G2.has_edge(*new2):
            continue

        # ensure the old edges still exist
        if not G2.has_edge(*old1) or not G2.has_edge(*old2):
            continue

        G2.remove_edge(*old1)
        G2.remove_edge(*old2)
        G2.add_edge(*new1)
        G2.add_edge(*new2)

        edges[i] = new1
        edges[j] = new2

    return G2

def one_degree_score(_):
    G_rand = randomize_G(G_valid)
    return nx.degree_assortativity_coefficient(G_rand)
        
# if __name__ == "__main__":
#     with Pool(cpu_count()) as pool:
#         Assortativity_scores = list(
#             tqdm(pool.imap(one_score, range(100)), total=100)
#         )
        
# if __name__ == "__main__":
#     with Pool(cpu_count()) as pool:
#         Assortativity_scores = list(pool.map(one_score, range(10)))

#     print(json.dumps(Assortativity_scores))

if __name__ == "__main__":
    with Pool(8) as pool:
        degree_scores = pool.map(one_degree_score, range(300))

    pd.Series(degree_scores).to_csv("assortativity_Degree_scores.csv", index=False)
    