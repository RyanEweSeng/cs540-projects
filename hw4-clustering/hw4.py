import csv
import numpy as np
from scipy.cluster.hierarchy import linkage

np.set_printoptions(suppress='True')

def load_data(filepath):
    file = open(filepath, 'r')
    reader = csv.DictReader(file)
    pokemon_list = list()

    for dict in reader:
        pokemon_list.append(dict)

    for dict in pokemon_list:
        del dict['#']
        del dict['Name']
        del dict['Type 1']
        del dict['Type 2']
        del dict['Total']
        del dict['Generation']
        del dict['Legendary']

    return pokemon_list

def calc_features(row):
    x1 = np.int64(row['Attack'])
    x2 = np.int64(row['Sp. Atk'])
    x3 = np.int64(row['Speed'])
    x4 = np.int64(row['Defense'])
    x5 = np.int64(row['Sp. Def'])
    x6 = np.int64(row['HP'])

    features = np.array([x1, x2, x3, x4, x5, x6])

    return features

def hac(features):
    n = len(features)
    Z = []

    # number the initial clusters from 0 to n - 1, those are their original cluster numbers
    # any new clusters we gain (from merging) are appended and given a respective number
    cluster_list = []
    for i in range(len(features)):
        pair = [i, [features[i]]]
        cluster_list.append(pair)
    curr_idx = len(features)

    # we keep merging clusters from our initial group of clusters i.e. there must always be at least 2 clusters present to merge
    while len(cluster_list) >= 2:
        idx1 = -1
        idx2 = -1
        complete_linkage_dist = np.inf
        cluster_size = -1

        # determine which clusters to merge: choose pair with smallest complete linkage distance
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                dist = calc_dist_from_norm(cluster_list[i][1], cluster_list[j][1])
                if dist < complete_linkage_dist:
                    idx1 = cluster_list[i][0]
                    idx2 = cluster_list[j][0]
                    complete_linkage_dist = dist
                elif dist == complete_linkage_dist:
                    if cluster_list[i][0] < idx1 or cluster_list[j][0] < idx2:
                        idx1 = cluster_list[i][0]
                        idx2 = cluster_list[j][0]
                        complete_linkage_dist = dist
                    else:
                        continue


        # merge the two chosen clusters
        cluster1 = helper_search_cluster(cluster_list, idx1)
        cluster2 = helper_search_cluster(cluster_list, idx2)
        merged_cluster = []
        for p in cluster1:
            merged_cluster.append(p)
        for p in cluster2:
            merged_cluster.append(p)

        # add cluster to our list of clusters
        new_pair = [curr_idx, merged_cluster]
        cluster_list.append(new_pair)
        curr_idx += 1

        # add cluster info to our result
        cluster_size = len(merged_cluster)
        info = [idx1, idx2, complete_linkage_dist, cluster_size]
        Z.append(info)

        # remove the merged clusters
        helper_delete_cluster(cluster_list, idx1)
        helper_delete_cluster(cluster_list, idx2)
        
    return np.array(Z, dtype=np.float64)

def calc_dist_from_norm(c1, c2):
    d = np.inf
    for p1 in c1:
        for p2 in c2:
            curr_d = np.linalg.norm(p1 - p2)
            if curr_d < d: d = curr_d

    return d

def helper_search_cluster(clusters, idx):
    for c in clusters:
        if c[0] == idx:
            return c[1]

    return null

def helper_delete_cluster(clusters, idx):
    for c in clusters:
        if c[0] == idx:
            clusters.remove(c)
            break

def imshow_haz(Z):
    pass

if __name__ == "__main__":
    data = load_data("Pokemon.csv")
    print(data)
    print("return type: ",type(data)) # expect list
    print("list element type: ", type(data[0])) # expect dict
    print()

    feature = calc_features(data[0])
    print(feature)
    print("return type: ",type(feature)) # expect numpy array
    print("array shape: ",feature.shape) # expect (6,0)
    print("array element type: ",type(feature[0])) # expect int64
    print()

    features = []
    for i in range(100):
        features.append(calc_features(data[i]))
    out_Z = hac(features)
    print(out_Z)
    print("return type: ",type(out_Z)) # expect numpy array
    print("array shape: ",out_Z.shape) # expect (n-1, 4)
    print("array element type: ",type(out_Z[0][0])) # expect float
    print()

    out_hac = linkage(features)
    print(out_hac)

