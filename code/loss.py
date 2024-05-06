import numpy as np
import torch


class DiffDP(torch.nn.Module):
    def __init__(self):
        super(DiffDP, self).__init__()

    def forward(self, y_pred, s):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        return reg_loss



class DRAlign(torch.nn.Module):
    def __init__(self):
        super(DRAlign, self).__init__()
        self.criterion = torch.nn.BCELoss()
    
    def forward(self, y_pred, s, y_true, model):
        y_pred, s, y_true = y_pred.reshape(-1), s.reshape(-1), y_true.reshape(-1)
        y0, y1 = y_pred[s==0], y_pred[s==1]
        gt0, gt1 = y_true[s==0], y_true[s==1]
        dp_loss = torch.abs(torch.mean(y0) - torch.mean(y1))

        loss_c_0 = self.criterion(y0, gt0)
        env_grad_0 = torch.autograd.grad(loss_c_0, model.parameters(), create_graph=True)
        loss_c_1 = self.criterion(y1, gt1)
        env_grad_1 = torch.autograd.grad(loss_c_1, model.parameters(), create_graph=True)

        loss_reg_s = 0
        for g0, g1, (name,para) in zip(env_grad_0, env_grad_1, model.named_parameters()):
            if "weight" in name and "fc" not in name and "head" not in name:
                nunits = para.shape[0]
                importance0 = (para*g0).pow(2).view(nunits,-1).sum(dim=1)
                importance1 = (para*g1).pow(2).view(nunits,-1).sum(dim=1)
                # if lam == 0:
                #     importance0 = importance0.detach()
                #     importance1 = importance1.detach()
                loss_reg_s += torch.cosine_similarity(importance0, importance1, dim=0)
        
        return dp_loss, loss_reg_s



class Mixup(torch.nn.Module):
    def __init__(self):
        super(Mixup, self).__init__()

    def forward(self, batch_x_0, batch_x_1, model):
        alpha = 1
        gamma = np.random.beta(alpha, alpha)

        batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
        batch_x_mix = batch_x_mix.requires_grad_(True)

        _, output = model(batch_x_mix)

        # gradient regularization
        gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

        batch_x_d = batch_x_1 - batch_x_0
        grad_inn = (gradx * batch_x_d).sum(1)
        E_grad = grad_inn.mean(0)
        reg_loss = torch.abs(E_grad)
        
        return reg_loss



class Manifold_Mixup(torch.nn.Module):
    def __init__(self):
        super(Manifold_Mixup, self).__init__()

    def forward(self, feat_0, feat_1, model):
        alpha = 1
        gamma = np.random.beta(alpha, alpha)

        # Manifold Mixup
        inputs_mix = feat_0 * gamma + feat_1 * (1 - gamma)
        inputs_mix = inputs_mix.requires_grad_(True)
        ops = model.linear(inputs_mix).sum()

        # Smoothness Regularization
        gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
        x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
        grad_inn = (gradx * x_d).sum(1).view(-1)
        loss_grad = torch.abs(grad_inn.mean())

        return loss_grad



class CDFdpTS(torch.nn.Module):
    def __init__(self, temperature):
        super(CDFdpTS, self).__init__()
        self.temperature = temperature
    
    def forward(self, y_pred, s):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        probing_points = torch.linspace(0, 1, steps=101).reshape(-1,1).to(y_pred.device)
        diff0 = probing_points - y_pred[s==0]
        diff1 = probing_points - y_pred[s==1]
        diff0 = torch.sigmoid(self.temperature * diff0)
        diff1 = torch.sigmoid(self.temperature * diff1)
        cdf0 = torch.mean(diff0, axis=1)
        cdf1 = torch.mean(diff1, axis=1)
        delta_ecdf = torch.abs(cdf0 - cdf1)
        reg_loss = torch.trapezoid(delta_ecdf, probing_points.reshape(-1))
        return reg_loss



class MaxCDFdp(torch.nn.Module):
    def __init__(self, temperature):
        super(MaxCDFdp, self).__init__()
        self.temperature = temperature
    
    def forward(self, y_pred, s):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        probing_points = torch.linspace(torch.min(y_pred).item(), torch.max(y_pred).item(), steps=100).reshape(-1,1).to(y_pred.device)
        diff0 = probing_points - y_pred[s==0]
        diff1 = probing_points - y_pred[s==1]
        diff0 = torch.sigmoid(self.temperature * diff0)
        diff1 = torch.sigmoid(self.temperature * diff1)
        cdf0 = torch.mean(diff0, axis=1)
        cdf1 = torch.mean(diff1, axis=1)
        delta_ecdf = torch.abs(cdf0 - cdf1)

        return torch.max(delta_ecdf)



class MaxDiffSolver(torch.nn.Module):
    def __init__(self, temperature, posratio):
        super(MaxDiffSolver, self).__init__()
        self.temperature = temperature
        self.ymax = torch.nn.Parameter(torch.FloatTensor([posratio]))

    def initialize(self, y):
        torch.nn.init.constant_(self.ymax.data, y)
    
    def forward(self, y_pred, s):
        diff0 = torch.sigmoid(self.temperature*(self.ymax-y_pred[s==0]))
        diff1 = torch.sigmoid(self.temperature*(self.ymax-y_pred[s==1]))
        delta_ecdf = torch.abs(torch.mean(diff0) - torch.mean(diff1))
        return -delta_ecdf
