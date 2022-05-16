import numpy as np

from torch.utils.data import Dataset

class ComparatorDataset(Dataset):
    def __init__(self, transforms = None) -> None:
        super().__init__()
        self.transforms = transforms

        # Total number of dataset = 10,000
        dataset_size = 1000
        self.inputs = np.zeros((dataset_size, 2))
                
        a = np.linspace(0.0, 1.0, 256)
        
        # self.inputs[:,0] = np.random.choice(a, dataset_size)
        self.inputs[:, 0] = np.random.choice(a, dataset_size) # 0.5
        self.inputs[:, 1] = np.random.choice(a, dataset_size)

        self.labels = np.zeros((self.inputs.shape[0], 1))
        for i in range(self.labels.shape[0]):
            if self.inputs[i, 0] >= self.inputs[i, 1]:
                self.labels[i] = 1.0

        # for i in range(self.inputs.shape[0]):
        #     if np.abs(self.inputs[i, 0] - self.inputs[i, 1]) <= 0.1:
        #         if self.labels[i] == 1.0:
        #             self.inputs[i, 0] += 0.1
        #         else:
        #             self.inputs[i, 1] += 0.1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        sample = (self.inputs[index,:], self.labels[index])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    training_data = ComparatorDataset(transforms=None)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    data = train_features[0]
    label = train_labels[0]

    print("Data is: ", data, "Corresponding label is: ", label)

