import numpy as np

def extract_trainval(L, ratios=[2, 3, 4]):
    ratios = sorted(ratios)

    vals = [i for i in range(9, L, 10)]
    trains = list(set(range(L)) - set(vals))
    ids = [trains]
    for r in ratios:
        trains_tmp = [trains[i] for i in range(0, len(trains), r)]
        ids.append(trains_tmp)
    return ids, vals
    

if __name__ == "__main__":
    L = 170
    trains, vals = extract_trainval(L)
    print(trains)
    print(vals)