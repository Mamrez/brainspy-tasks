import torch
from torch.utils.data import DataLoader

import numpy as np

class FCLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes, drop_out_prob) -> None:
        super(FCLayer, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.relu = torch.nn.ReLU()

        self.hidden_layer = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout = torch.nn.Dropout(p=drop_out_prob)
        self.relu_2 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(hidden_layer_size, num_classes)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)

        out = self.hidden_layer(out)
        out = self.relu_2(out)    
        out = self.dropout(out)  

        out = self.fc2(out)
        out = torch.log_softmax(out, dim=1)

        return out

def test(
    model,
    dataset,
    device
    ):

    test_dataloader = DataLoader(
        dataset,
        batch_size= 1,
        shuffle= True,
    )
    print("Length of dataset: ", len(test_dataloader))
    correct, total = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print("Test accuracy: ", 100. * correct / total)

def NNmodel(
    NNtype= 'LinearLayer',
    down_sample_no = 512,
    hidden_layer_size = 512,
    num_classes = 10,
    dropout_prob = 0.1
):
    if NNtype == 'FC':
        tmp = FCLayer(down_sample_no, hidden_layer_size, num_classes, dropout_prob)
    
    return tmp
 
if __name__ == '__main__':
 
    down_sample_no = 256

    hidden_layer_size = 128

    batch_size = 128

    num_classes = 10

    classification_layer = NNmodel(
        NNtype= 'FC', 
        down_sample_no= down_sample_no,
        hidden_layer_size = hidden_layer_size,
        num_classes= 10,
        dropout_prob= 0.5,
    )

    print("Number of learnable parameters are: ", sum(p.numel() for p in classification_layer.parameters()))

    classification_layer.load_state_dict(
        torch.load(
            "saved_model.pt", map_location='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
    )
    model = classification_layer.to(
        device=torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
    )
    model.eval()
    test_dataset = np.load("test_set.npy", allow_pickle=True)

    test(
        model,
        test_dataset,
        device=  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
