import csv
import numpy as np

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
    pass

def imshow_haz(Z):
    pass

if __name__ == "__main__":
    res = load_data("Pokemon.csv")
    print("return type: ",type(res)) # expect list
    print("list element type: ", type(res[0])) # expect dict
    print("\n")

    feature = calc_features(res[0])
    print("return type: ",type(feature)) # expect numpy array
    print("array shape: ",feature.shape) # expect (6,0)
    print("array element type: ",type(feature[0])) # expect int64
    print("\n")
