from torch.utils.data import Dataset


class TrialDataset(Dataset):

    def __init__(self, channel_stimulus, channel_response, stimulus_labels, response_labels, subjects):
        self.channel_stimulus = channel_stimulus
        self.channel_response = channel_response

        self.stimulus_labels = stimulus_labels
        self.response_labels = response_labels

        self.subjects = subjects

    def __len__(self):
        return len(self.channel_stimulus[0])

    def __getitem__(self, item):
        result = []
        for index in range(len(self.channel_stimulus)):
            result.append(self.channel_stimulus[index][item])
            result.append(self.channel_response[index][item])
        result.append(self.stimulus_labels[item])
        result.append(self.response_labels[item])
        result.append(self.subjects[item])
        return result


class DatasetForClassificationStatistics(Dataset):
    def __init__(self, channel_stimulus, channel_response, stimulus_labels, response_labels, subjects, trial_index):
        self.channel_stimulus = channel_stimulus
        self.channel_response = channel_response

        self.stimulus_labels = stimulus_labels
        self.response_labels = response_labels

        self.subjects = subjects

        self.trial_index = trial_index

    def __len__(self):
        return len(self.channel_stimulus[0])

    def __getitem__(self, item):
        result = []
        for index in range(len(self.channel_stimulus)):
            result.append(self.channel_stimulus[index][item])
            result.append(self.channel_response[index][item])
        result.append(self.stimulus_labels[item])
        result.append(self.response_labels[item])
        result.append(self.subjects[item])
        result.append(self.trial_index[item])
        return result
