import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm

import numpy as np

from brainspy.processors.simulation.processor import SurrogateModel

class smg_activation_function(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        training_data = torch.load(
            "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/learnable_nonlinearity/smg/training_data.pt"
        )

        self.smg = torch.compile(SurrogateModel(
            model_structure     = training_data['info']['model_structure'],
            model_state_dict    = training_data['model_state_dict']
        ))
        
        # defining trainable parameters
        # 1 -> input
        # 6 -> trainable control voltges
        # 1 -> output
        self.control_voltages = nn.parameter.Parameter(
                torch.tensor([0., 0., 0., 0., 0., 0.])
        )
        
        self.control_voltages.requires_grad = True
    
    def forward(self, x):
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                x[i][j] = self.smg((torch.cat((x[i][j].reshape(1), self.control_voltages))))
        
        return x
        

class MLP_trainable_nonlinearity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(484, 240)
        self.nonlinearity_1 = smg_activation_function()
        self.fc2 = nn.Linear(240, 10)
        self.nonlinearity_2 = smg_activation_function()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(x))

        x = self.nonlinearity_1(self.fc1(x))
        out = self.nonlinearity_2(self.fc2(x))

        return F.log_softmax(out, dim=1)
        
def load_smg(
        path = None
):
    if path == None:
        training_data = torch.load(
            "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/learnable_nonlinearity/smg/training_data.pt"
        )
    else:
        training_data = torch.load(
            path
        )

    smg = SurrogateModel(
        model_structure     = training_data['info']['model_structure'],
        model_state_dict    = training_data['model_state_dict']
    )

    return smg

def train(
        model,
        num_epochs,
        weight_decay,
        train_loader,
        test_loader,
        device,
        batch_size
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 0.001
    )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), 
    #     weight_decay    = weight_decay
    # )

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr          = 0.01,
    #     steps_per_epoch = int(len(train_loader)),
    #     epochs          = num_epochs,
    #     anneal_strategy = 'cos',
    #     cycle_momentum  = True
    # )
    # model.train()
    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            # i = 0
            for i, (data, target) in enumerate(tepoch):
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Epoch {epoch}")
                
                outputs = torch.squeeze(model(data))
                loss = loss_fn(outputs, target)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                # scheduler.step()
                
                current_loss += loss.item()

                if i % batch_size  == batch_size - 1:
                    current_loss = 0.

                tepoch.set_postfix(loss=current_loss)

        # if epoch == num_epochs - 1:
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for t, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                outputs = torch.squeeze(model(data))
                _, predicted = torch.max(outputs, 1)
                total += data.size(0)
                correct += (predicted == target).sum().item()

        print("Test accuracy: ", 100 * correct / total)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root  = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/mnist",
            train = True,
            download= True,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                torchvision.transforms.CenterCrop(22)
            ]),
        ),
        batch_size  = 32,
        shuffle     = True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/mnist",
            train= False,
            download= True,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                torchvision.transforms.CenterCrop(22)
            ]),
        ),
        batch_size  = 50,
        shuffle     = True
    )

    model = MLP_trainable_nonlinearity()
    model.to(device)

    train(
        model               = model,
        num_epochs          = 100,
        weight_decay        = 1e-5,
        train_loader        = train_loader,
        test_loader         = test_loader,
        device              = device,
        batch_size          = 32
    )
    
    pass