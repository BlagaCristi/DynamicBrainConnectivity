import numpy as np
from torch.utils.data import Dataset


class GraphWavenetDataset(Dataset):

    def __init__(self, input_sequence, output_sequence):
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence

    def __len__(self):
        return len(self.input_sequence)

    def __getitem__(self, item):
        return np.array([self.input_sequence[item]]), np.array([self.output_sequence[item]])
