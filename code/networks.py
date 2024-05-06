import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class resnet18_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=512):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x

class GradReverse(Function):
    """
    borrwed from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)



class MLP(nn.Module):
    def __init__(self, n_features, mlp_layers= [512, 256, 64], p_dropout=0.2, num_classes=1):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.mlp_layers = [n_features] + mlp_layers
        self.p_dropout = p_dropout

        self.network = nn.ModuleList( [nn.Linear(i, o) for i, o in zip( self.mlp_layers[:-1], self.mlp_layers[1:])] )
        self.head = nn.Linear(self.mlp_layers[-1], num_classes)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        h = x
        x = self.head(x)
        return h, torch.sigmoid(x)



class AdvDebiasing(nn.Module):
    """
    modified from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Multi-layer perceptron with adversarial training for fairness.
    """

    def __init__(self, n_features, num_classes=1, hidden_layers=[60], adversary_layers=[50]):
        super(AdvDebiasing, self).__init__()
        self.input_dim = n_features
        self.num_classes = num_classes
        self.num_hidden_layers = len(hidden_layers)
        self.num_neurons = [self.input_dim] + hidden_layers

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])for i in range(self.num_hidden_layers)])
        
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], self.num_classes)

        # Parameter of the adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + adversary_layers
        self.num_adversaries_layers = len(adversary_layers)
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 1)

    def forward(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))

        # Classification probability.
        logprobs = torch.sigmoid(self.softmax(h_relu))

        # Adversary classification component.
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))

        cls = torch.sigmoid(self.sensitive_cls(h_relu))
        return logprobs, cls



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



import torch
from torch import nn
import torch.nn.functional as F


class AdvImage(nn.Module):
    # def __init__(self, n_features, num_classes=1, hidden_layers=[60], adversary_layers=[50]):
    def __init__(self, pretrained, n_hidden=512, num_classes=1, adversary_layers=[50]):
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(512, 1)

        # Parameter of the adversary classification layer.
        self.num_adversaries = [n_hidden] + adversary_layers
        self.num_adversaries_layers = len(adversary_layers)
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 1)

    def forward(self, x):

        h = self.resnet(x)
        x = self.fc(h)
        x = torch.sigmoid(x)

        h_relu = F.relu(h)
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))

        cls = torch.sigmoid(self.sensitive_cls(h_relu))
        return x, cls



class resnet_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=512):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def linear(self, x):
        x = self.avg(x.view(-1, 512, 2, 2)).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.linear(x)
        return h, x
