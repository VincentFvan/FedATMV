# utils/ShakeSpeare_reduce.py

import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from utils.language_utils import word_to_indices, letter_to_vec

def read_dir(data_dir):
    """Reads all .json files in a directory."""
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])
    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    """
    Parses data from train and test directories.
    Returns:
      clients: A list of client IDs.
      groups: A list of group IDs.
      train_data: A dictionary of training data.
      test_data: A dictionary of testing data.
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    assert train_clients == test_clients
    assert train_groups == test_groups
    return train_clients, train_groups, train_data, test_data

class ShakeSpeare(Dataset):
    """
    A PyTorch Dataset for the Shakespeare dataset.
    This version reduces the number of samples per client by a given factor.
    """
    def __init__(self, train=True, reduce_factor=0.5):
        """
        Args:
          - train (bool): If True, creates the training dataset, otherwise the test dataset.
          - reduce_factor (float): The proportion of samples to keep for each class within each client.
        """
        super(ShakeSpeare, self).__init__()
        train_clients, _, train_data_temp, test_data_temp = read_data(
            "./data/shakespeare/train", "./data/shakespeare/test"
        )
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x, train_data_y = [], []
            
            # Process data for each client individually
            for i, client_id in enumerate(train_clients):
                self.dic_users[i] = set()
                cur_x = train_data_temp[client_id]["x"]
                cur_y = train_data_temp[client_id]["y"]

                # Group data by class for the current client
                class_indices = defaultdict(list)
                for j, label in enumerate(cur_y):
                    class_indices[label].append(j)

                # Sample a 'reduce_factor' portion of data for each class
                selected_indices = []
                for label, indices in class_indices.items():
                    count_to_keep = int(len(indices) * reduce_factor)
                    if count_to_keep == 0 and len(indices) > 0:
                        count_to_keep = 1  # Keep at least one sample
                    selected = np.random.choice(indices, count_to_keep, replace=False)
                    selected_indices.extend(selected)
                selected_indices.sort()

                # Append selected data and record global indices for the client
                for j in selected_indices:
                    current_idx = len(train_data_x)
                    self.dic_users[i].add(current_idx)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            # Use all available test data
            test_data_x, test_data_y = [], []
            for client_id in train_clients:
                cur_x = test_data_temp[client_id]["x"]
                cur_y = test_data_temp[client_id]["y"]
                test_data_x.extend(cur_x)
                test_data_y.extend(cur_y)
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        indices = torch.LongTensor(np.array(indices))
        return indices, target

    def get_client_dic(self):
        """Returns the dictionary mapping client IDs to their data indices."""
        if self.train:
            return self.dic_users
        else:
            raise RuntimeError("Test dataset does not have dic_users!")