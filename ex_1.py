import sys
import math
import numpy as np
import scipy.io.wavfile


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)  # Data read from the text file.


    # blah blah blah...

    # scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))

def k_Means_Algorithm(sample, centroids):
    count = 0  # Counting the iterations of the algorithm.
    con = False  # Checking if we've reached convergence.
    

if __name__ == "__main__":
    main()
