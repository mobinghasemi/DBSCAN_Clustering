import math


class DBSCAN:
        def __init__(self,data,eps,min_samples):
            self.DB = self.dbscan(data,eps,min_samples)

        @staticmethod
        def euclidean_distance(point1, point2):

            distance = 0
            for i in range(len(point1)):
                distance += (point1[i] - point2[i]) ** 2
            return math.sqrt(distance)


        @staticmethod
        def get_neighbors(data, point_index, eps):

            neighbors = []
            for i in range(len(data)):
                if DBSCAN.euclidean_distance(data[point_index], data[i]) <= eps:
                    neighbors.append(i)
            return neighbors


        @staticmethod
        def dbscan(data, eps, min_samples):
            labels = [-1] * len(data)  # Initialize all points as noise (-1)
            cluster_id = 0
            for i in range(len(data)):
                if labels[i] != -1:  # Skip points that have already been assigned to a cluster
                    continue
                neighbors = DBSCAN.get_neighbors(data, i, eps)
                if len(neighbors) < min_samples:
                    labels[i] = -1
                    continue
                cluster_id += 1
                labels[i] = cluster_id
                j = 0
                while j < len(neighbors):
                    neighbor_index = neighbors[j]
                    if labels[neighbor_index] == -1:
                        labels[neighbor_index] = cluster_id
                        neighbor_neighbors = DBSCAN.get_neighbors(data, neighbor_index, eps)
                        if len(neighbor_neighbors) >= min_samples:
                             neighbors += neighbor_neighbors
                    j += 1
            return labels, cluster_id



