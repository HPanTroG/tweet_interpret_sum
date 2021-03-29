from itertools import combinations
import numpy as np


class LSH:
    # map similar sentences in terms of cosine similarity to same buckets
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self, num_bits, seed=45):
        dim = self.data.shape[1]
        np.random.seed(seed)
        random_vectors = np.random.randn(dim, num_bits)

        powers_of_two = 1 << np.arange(num_bits - 1, -1, -1)  # [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        table = {}

        # partition data points into bins
        bin_index_bits = (self.data.dot(random_vectors) >= 0)
        # encode bin index bits into intergers
        bin_indices = bin_index_bits.dot(powers_of_two)

        print(bin_indices.shape)

        # update table, table[i]: list of document ids with bin_index = i
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                table[bin_index] = []
            table[bin_index].append(data_index)

        self.model = {'bin_indices': set(bin_indices), 'table': table,
                      'random_vectors': random_vectors, 'num_bits': num_bits}

    def extract_nearby_bins(self, max_search_radius=1):
        buckets = []
        power_of_two = 1 << np.arange(self.model['num_bits'] - 1, -1, -1)

        for binx in self.model['bin_indices']:
            bin_in_binary = '{0:b}'.format(binx)
            bin_in_binary = [0] * (self.model['num_bits'] - len(bin_in_binary)) + [int(b) for b in bin_in_binary]

            candidates = self.model['table'][binx].copy()
            for radius in range(1, max_search_radius + 1):
                for different_bits in combinations(range(self.model['num_bits']), radius):
                    alternative_bits = bin_in_binary.copy()
                    for i in different_bits:
                        alternative_bits[i] = 1-alternative_bits[i]
                    # convert nit vector to interger
                    nearby_bin = np.array(alternative_bits).dot(power_of_two)

                    if nearby_bin in self.model['table']:
                        candidates += self.model['table'][nearby_bin]

            if len(candidates) > 1:
                buckets.append(candidates)

        return buckets
