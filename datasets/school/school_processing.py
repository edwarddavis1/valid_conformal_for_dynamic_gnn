import numpy as np


def get_school_data(dir):
    window = 60 * 60

    t_start = (8 * 60 + 30) * 60
    t_end = (17 * 60 + 30) * 60

    T = int((t_end - t_start) // window)
    print(f"Number of time windows: {T}")

    fname = dir + "/ia-primary-school-proximity-attr.edges"
    file = open(fname)

    label_dict = {
        "1A": 0,
        "1B": 1,
        "2A": 2,
        "2B": 3,
        "3A": 4,
        "3B": 5,
        "4A": 6,
        "4B": 7,
        "5A": 8,
        "5B": 9,
        "Teachers": 10,
    }
    nodes = []
    node_labels = []
    edge_tuples = []

    for line in file:
        node_i, node_j, time, id_i, id_j = line.strip("\n").split(",")

        if int(time) > 24 * 60 * 60:
            continue

        if node_i not in nodes:
            nodes.append(node_i)
            node_labels.append(label_dict[id_i])

        if node_j not in nodes:
            nodes.append(node_j)
            node_labels.append(label_dict[id_j])

        t = (int(time) - t_start) // window
        edge_tuples.append([t, node_i, node_j])

    edge_tuples = np.unique(edge_tuples, axis=0)
    nodes = np.array(nodes)

    n = len(nodes)
    print(f"Number of nodes: {n}")

    node_dict = dict(zip(nodes[np.argsort(node_labels)], range(n)))
    node_labels = np.sort(node_labels)

    As = np.zeros((T, n, n))

    for m in range(len(edge_tuples)):
        t, i, j = edge_tuples[m]
        As[int(t), node_dict[i], node_dict[j]] = 1
        As[int(t), node_dict[j], node_dict[i]] = 1

    return As, node_labels
