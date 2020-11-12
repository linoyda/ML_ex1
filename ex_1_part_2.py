import sys
import math
import numpy as np
from numpy import random
import scipy.io.wavfile
import matplotlib.pyplot as plt


def main():
    count = 0  # Counting the iterations of the algorithm.
    con = False  # Checking if we've reached convergence.
    total_costs_list = []
    sample = sys.argv[1]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())

    min_values, max_values = np.amin(x, axis=0), np.amax(x, axis=0)
    centroids = get_random_centroids(min_values[0], max_values[0], min_values[1], max_values[1])
    out_file = open("output02.txt", "x")

    while not con and count < 10:
        # todo continue from here...
        prev_centroids = centroids
        new_centroids, curr_costs = k_means_algorithm(centroids, x)
        total_costs_list.extend(curr_costs)

        # Writing the current iteration's centroids into the out_file
        out_file.write(f"[iter {count}]:{', '.join([str(j) for j in new_centroids])}\n")
        count += 1
        # Check if the previous list of centroids consists of the same elements as the new one
        if (prev_centroids == new_centroids).all():
            con = True
        else:
            centroids = new_centroids
    draw_graph_according_to_costs(total_costs_list)
    # scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))


def determine_cost_as_sum_according_to_dictionary(dictionary, centroids):
    res = 0
    for cluster in dictionary:
        for point in dictionary[cluster]:
            res += distance_metric(centroids[cluster], point)
    return res


def draw_graph_according_to_costs(costs_list):
    x_range = range(len(costs_list))
    plt.plot(x_range, costs_list)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
    print("finished")


def get_random_centroids(min_x, max_x, min_y, max_y):
    random_centroids = []
    required_k = 16  # 2 / 4 / 8 / 16
    for iteration in range(required_k):
        x_randomized = random.randint(min_x, max_x)
        y_randomized = random.randint(min_y, max_y)
        random_centroids.append([x_randomized, y_randomized])
    return random_centroids


def k_means_algorithm(centroids, x):
    costs_list = []
    dict_of_points = determine_clusters_according_to_closest_centroid(x, centroids)
    # change -- calculate the sum of distances, a.k.a the COST
    costs_list.append(determine_cost_as_sum_according_to_dictionary(dict_of_points, centroids))
    new_centroids = determine_k_new_centroids_according_to_dict(dict_of_points, centroids)
    return new_centroids, costs_list


def distance_metric(point_a, point_b):
    # Inspiration: https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25
    # return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def determine_clusters_according_to_closest_centroid(clusters_arr, centroids):
    dictionary = dict()
    for centroid_index in range(len(centroids)):
        dictionary.setdefault(centroid_index, [])  # initialize a new line. key: cluster index. value: empty set
    for i in clusters_arr:  # for each point in x arr
        distances_list = []
        for curr_centroid in centroids:
            distances_list.append(distance_metric(curr_centroid, i))
        # min_distance = min(distances_list)
        min_distance = distances_list.index(np.amin(distances_list))  # The centroid the point is assigned to

        # dictionary.setdefault(min_distance, [])  # initialize a new line. key: cluster. value: empty set
        dictionary[min_distance].append(i)  # Add the current point as a value of the current key.

    # if amount of keys of dictionary != amount of centroids -- add the remaining centroids to the dictionary.
    return dictionary


def determine_k_new_centroids_according_to_dict(dictionary, centroids):
    # This function calculates the means, the centroids of each cluster.
    # The keys of the dictionary are the indexes of the centroids.
    # Each value is a list of all points assigned to the cluster
    new_centroids = []

    for cluster in dictionary:  # for each cluster
        # count how many points are in each cluster BEFORE CHANGE
        points_assigned_to_cluster = len(dictionary[cluster])
        if points_assigned_to_cluster == 0:
            # don't divide in zero! Just add the previous centroid...
            new_centroids.append(centroids[cluster])
        else:
            total_arr = np.sum(dictionary[cluster], axis=0)
            # total_arr[0] - the sum of all x values of points.
            # total_arr[1] - the sum of all y values of points.
            new_centroid_x = (total_arr[0] / points_assigned_to_cluster).round()
            new_centroid_y = (total_arr[1] / points_assigned_to_cluster).round()

            new_centroids.append((new_centroid_x, new_centroid_y))
    new_centroids = np.asarray(new_centroids)
    return new_centroids


if __name__ == "__main__":
    main()
