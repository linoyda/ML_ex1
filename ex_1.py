import sys
import math
import numpy as np
import scipy.io.wavfile


def main():
    count = 0  # Counting the iterations of the algorithm.
    con = False  # Checking if we've reached convergence.

    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)  # Loading the centroids to an array of points

    # blah blah blah...
    out_file = open("output.txt", "x")
    while not con and count < 30:
        new_centroids = []
        k_means_algorithm(out_file, sample, centroids)
        count += 1 # Performed one more iteration
        # TODO get new_centroids from algorithm.


    # Writing the current iteration's centroids into the out_file
    out_file.write(f"[iter {count}]:{', '.join([str(j) for j in new_centroids])}\n")
    # scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))


def k_means_algorithm(out_file, sample, centroids):



def distance_metric(point_a, point_b):
    # Inspiration: https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25
    return (sum((point_a - point_b) ** 2)) ** 0.5

def determine_clusters_according_to_centroids(clusters, centroids):
    new_clusters = []
    for index in range(clusters.shape[0]):
        distances_set = []
        for current_centroid in centroids:
            distances_set.append(distance_metric(current_centroid, clusters[index]))
        # If the current_distance is the minimum of all distances between the current centroid and
        # the current cluster, add it to the new set of clusters.
        cluster = [z for z, current_distance in enumerate(distances_set) if current_distance == min(distances_set)]
        new_clusters.append(cluster[0])
    return new_clusters

def determine_new_centroids_according_to_means(cluster_array, clusters):
    new_centroids = []
    # TODO Continue function
    
if __name__ == "__main__":
    main()
