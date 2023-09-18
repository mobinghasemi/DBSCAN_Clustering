from itertools import count

import pandas as pd

from dbscan import DBSCAN
from matplotlib import pyplot as plt, cm
from evaluation_criteria import davies_bouldin_index
name = 'male'
file = f'datasetnew_{name}1.csv'

def open_csv_file():
    data = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith(f'norm_avg_{name}'):  # Skip header row
                continue
            fields = line.strip().split(',')
            point = tuple([float(field) for field in fields])
            data.append(point)
    return data



def show_on_plot(labels, data):
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta',
              'lime', 'indigo', 'teal', 'silver', 'maroon', 'olive', 'navy', 'aqua', 'fuchsia',
              'crimson', 'darkgreen', 'darkblue', 'darkorange', 'darkviolet', 'gold', 'gray']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(data)):
        if labels[i] == -1:
            ax.scatter(data[i][0], data[i][1], data[i][2], color='black', s=20)
        else:
            ax.scatter(data[i][0], data[i][1], data[i][2], color=colors[labels[i] % len(colors)], s=20)

    plt.show()


if __name__ == "__main__":
    data = open_csv_file()
    labels, cluster_count = DBSCAN(data, eps=0.1, min_samples=1).DB
    print(f"clusters count: ",{cluster_count})
    # X = pd.read_csv('datasetnew_female.csv')
    print(f"davis: ",davies_bouldin_index(file,labels))
    counter = 0
    for i in labels:
        if i == -1:
            counter+=1
    print("count of outlier is: ",counter)


    show_on_plot(labels,data)
    # show_on_plot(labels, data)

