import sys
import math
import numpy as np
import scipy.io.wavfile


def main():
    count = 0  # Counting the iterations of the algorithm.
    con = False  # Checking if we've reached convergence.
    new_centroids = []
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)  # Loading the centroids to an array of points

    out_file = open("output.txt", "x")
    while not con and count < 30:
        prev_centroids = centroids
        new_centroids = k_means_algorithm(centroids, x)

        # Writing the current iteration's centroids into the out_file
        out_file.write(f"[iter {count}]:{', '.join([str(j) for j in new_centroids])}\n")
        count += 1
        # Check if the previous list of centroids consists of the same elements as the new one
        if (prev_centroids == new_centroids).all():
            con = True
        else:
            centroids = new_centroids
    # scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))


def k_means_algorithm(centroids, x):
    dict_of_points = determine_clusters_according_to_closest_centroid(x, centroids)
    new_centroids = determine_k_new_centroids_according_to_dict(dict_of_points, centroids)
    return new_centroids


def distance_metric(point_a, point_b):
    # Inspiration: https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


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
    # new_centroids = np.array([0])

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
