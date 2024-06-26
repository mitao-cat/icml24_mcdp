import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from tabulate import tabulate
from torchvision import transforms
from torch.utils.data import TensorDataset

from dataset import load_celeba_data
from utils import seed_everything, print_metrics, clear_lines, InfiniteDataLoader
from metrics import metric_evaluation
from networks import resnet_encoder
from loss import Manifold_Mixup

import pickle


tfms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def train_step(model, data, target, sensitive, scheduler, optimizer, clf_criterion, fair_criterion, lam, device, args=None):
    (X0,X1), (y0,y1), (s0,s1) = data, target, sensitive
    model.train()
    optimizer.zero_grad()
    h0, output0 = model(X0)
    h1, output1 = model(X1)

    clf_loss = clf_criterion(torch.cat([output0,output1]), torch.cat([y0,y1]))
    fair_loss = fair_criterion(h0, h1, model)
    loss = clf_loss + lam * fair_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model



def test(model, test_loader, clf_criterion, fair_criterion, lam, device, target_index = 31, sensitive_index = 20, prefix="test", args=None):
    with torch.no_grad():
        model.eval()
        target_hat_list = []
        target_list = []
        sensitive_list = []
        with torch.no_grad():
            for X, attr in test_loader:
                
                data = X.to( device )
                target = attr[:,target_index].to(  device )
                target = target.unsqueeze(1).type(torch.float)
                sensitive = attr[:,sensitive_index].to(  device )
                sensitive = sensitive.unsqueeze(1).type(torch.float)

                h, output = model(data)
                target_hat_list.append(output.cpu().numpy())
                target_list.append(target.cpu().numpy())
                sensitive_list.append(sensitive.cpu().numpy())

        target_hat_list = np.concatenate(target_hat_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        sensitive_list = np.concatenate(sensitive_list, axis=0)
        metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}", epsilon_list=args.epsilon_list, K=args.K)

    return metric




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/", help="Path to the dataset folder")
    parser.add_argument("--dataset", type=str, default="celeba", help="Choose a dataset from the available options: celeba")
    parser.add_argument("--model", type=str, default="mixup", help="Model type")
    parser.add_argument("--architecture", type=str, default="resnet18", help="the architecture of the model")
    parser.add_argument("--target_attr", type=str, default="Attractive", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="Gender", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--evaluation_metrics", type=str, default="ap,dp,abcc,mcdp0", help="Evaluation metrics separated by commas")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency")

    parser.add_argument("--lam", type=float, default=1)

    parser.add_argument("--num_training_steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="uuid", help="Experiment name")
    parser.add_argument("--save", type=int, default=0, help="Saving the well-trained model locally")
    parser.add_argument("--epsilon", type=float, default=0.01, help="epsilon for computing mcdp")
    parser.add_argument("--K", type=int, default=1, help="K for computing mcdp")


    # Parse the command-line arguments
    args = parser.parse_args()
    args.epsilon_list = [0.001,0.01,0.02,0.05,0.1] # you may manually specify epsilon here

    # Create a table of the arguments and their values
    table = tabulate([(k, v) for k, v in vars(args).items()], tablefmt='grid')
    print(table)


    seed_everything(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    root_dir = ".."
    if args.dataset == "celeba":
        X, attr = load_celeba_data(path= root_dir+"/datasets/celeba/raw", sensitive_attribute=args.sensitive_attr)

        X_np = np.stack( X["pixels"].to_list() )
        attr_np = attr.to_numpy()

        print( "X.shape: ", X.shape )
        print( "attr_np.shape: ", attr_np.shape )


        X_train, X_testvalid, attr_train, attr_testvalid = train_test_split(X_np, attr_np, test_size=0.2, stratify=attr_np, random_state=args.seed)
        X_test, X_val, attr_test, attr_val = train_test_split(X_testvalid, attr_testvalid, test_size=0.5, stratify=attr_testvalid, random_state=args.seed)

        print( "X_train.shape: ", X_train.shape )
        print( "X_val.shape: ", X_val.shape )
        print( "X_test.shape: ", X_test.shape )

        X_train, attr_train = torch.from_numpy(X_train).float(), torch.from_numpy(attr_train).float()
        X_val, attr_val = torch.from_numpy(X_val).float(), torch.from_numpy(attr_val).float()
        X_test, attr_test = torch.from_numpy(X_test).float(), torch.from_numpy(attr_test).float()


        train_dataset = TensorDataset(X_train, attr_train)
        val_dataset = TensorDataset(X_val, attr_val)
        test_dataset = TensorDataset(X_test, attr_test)

        if args.target_attr == "Smiling":
            target_index = 0
            print( "target_attr is Smiling!" )
        elif args.target_attr == "Wavy_Hair":
            target_index = 1
            print( "target_attr is Wavy_Hair!" )
            args.dataset = 'celeba-w'
        elif args.target_attr == "Attractive":
            target_index = 2
            print( "target_attr is Attractive!" )
            args.dataset = 'celeba-a'
        else:
            NotImplementedError("target_attr is not implemented!")


        if args.sensitive_attr == "Gender":
            sensitive_index = 3
            print( "sensitive_attr is Gender!" )
        elif args.sensitive_attr == "Young":
            sensitive_index = 4
            print( "sensitive_attr is Young!" )
        else:
            NotImplementedError("sensitive_attr is not implemented!")
        y = attr.iloc[:,target_index]
        s = attr.iloc[:,sensitive_index]

    dataset_stats = {
        "dataset": args.dataset,
        "num_classes": len(np.unique(y)),
        "num_sensitive": len(np.unique(s)),
        "num_samples": len(train_dataset)+len(val_dataset)+len(test_dataset),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "num_y1": (y.values == 1).sum(),
        "num_y0": (y.values == 0).sum(),
        "num_s1": (s.values == 1).sum(),
        "num_s0": (s.values == 0).sum(),
    }


    # Create the table using the tabulate function
    table = tabulate([(k, v) for k, v in dataset_stats.items()], tablefmt='grid')
    print(table)

    train_infinite_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.evaluation_batch_size, num_workers=4, drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.evaluation_batch_size, num_workers=4, drop_last=False, pin_memory=True)

    y_train, s_train = attr_train[:,target_index], attr_train[:,sensitive_index]
    X_train_0, y_train_0, s_train_0 = X_train[s_train==0], y_train[s_train==0], s_train[s_train==0]
    X_train_1, y_train_1, s_train_1 = X_train[s_train==1], y_train[s_train==1], s_train[s_train==1]
    y_train_0, s_train_0, y_train_1, s_train_1 = y_train_0.unsqueeze(1), s_train_0.unsqueeze(1), y_train_1.unsqueeze(1), s_train_1.unsqueeze(1)
    train_data_0, train_data_1 = TensorDataset(X_train_0, y_train_0, s_train_0), TensorDataset(X_train_1, y_train_1, s_train_1)
    train_infinite_loader_0 = InfiniteDataLoader(train_data_0, batch_size=int(args.batch_size/2), shuffle=True, drop_last=True)
    train_infinite_loader_1 = InfiniteDataLoader(train_data_1, batch_size=int(args.batch_size/2), shuffle=True, drop_last=True)

    net = resnet_encoder(pretrained=True).to(device)

    criterion = nn.BCELoss()
    fair_criterion = Manifold_Mixup()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    print(net)


    logs = []
    headers = ["Step(Val|Te)"] + args.evaluation_metrics.split(",")


    for step, ((X0, y0, s0), (X1, y1, s1)) in enumerate(zip(train_infinite_loader_0, train_infinite_loader_1)):
        if step >= args.num_training_steps:
            break
        X0, y0, s0, X1, y1, s1 = X0.to(device), y0.to(device), s0.to(device), X1.to(device), y1.to(device), s1.to(device)
        net = train_step(model=net, data=(X0,X1), target=(y0,y1), sensitive=(s0,s1), optimizer=optimizer, scheduler=scheduler,  clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, device=device)

        if step % args.log_freq == 0 or step == 1 or step == args.num_training_steps:

            val_metrics   = test(model=net, test_loader=val_loader, clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, target_index = target_index, sensitive_index = sensitive_index, device=device, prefix="val", args=args)
            test_metrics  =  test(model=net, test_loader=test_loader, clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, target_index = target_index, sensitive_index = sensitive_index, device=device, prefix="test", args=args)
            res_dict = {}
            res_dict.update(val_metrics)
            res_dict.update(test_metrics)

            # for printing
            if step % (args.log_freq*10) == 0:
                res = print_metrics(res_dict, args.evaluation_metrics, train=False)
                logs.append( [ step, *res] )
                if  step > 3:
                    clear_lines(len(logs)*2 + 1)
                table = tabulate(logs, headers=headers, tablefmt="grid", floatfmt="02.2f")
                print(table)


    with torch.no_grad():   # Final eval/test
        setting_dict = {'dataset':args.dataset, 'method':args.model, 'sensitive_attr':args.sensitive_attr, 'target_attr':args.target_attr, 'lam':args.lam, 'seed':args.seed}
        net.eval()

        target_hat_list, target_list, sensitive_list = [], [], []
        for X, attr in val_loader:   
            data = X.to( device )
            target = attr[:,target_index].to(  device )
            target = target.unsqueeze(1).type(torch.float)
            sensitive = attr[:,sensitive_index].to(  device )
            sensitive = sensitive.unsqueeze(1).type(torch.float)

            h, output = net(data)
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

        target_hat_list = np.concatenate(target_hat_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        sensitive_list = np.concatenate(sensitive_list, axis=0)

        metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, epsilon_list=args.epsilon_list, K=args.K)
        val_dict = {'ap':metric['/ap'], 'dp':metric['/dp'], 'dpe':metric['/dpe'], 'abcc':metric['/abcc'], 'mcdp_a':metric['/mcdp_a'], 'mcdp0':metric['/mcdp0']}



        target_hat_list, target_list, sensitive_list = [], [], []
        for X, attr in test_loader:   
            data = X.to( device )
            target = attr[:,target_index].to(  device )
            target = target.unsqueeze(1).type(torch.float)
            sensitive = attr[:,sensitive_index].to(  device )
            sensitive = sensitive.unsqueeze(1).type(torch.float)

            h, output = net(data)
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

        target_hat_list = np.concatenate(target_hat_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        sensitive_list = np.concatenate(sensitive_list, axis=0)

        metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, epsilon_list=args.epsilon_list, K=args.K)
        test_dict = {'ap':metric['/ap'], 'dp':metric['/dp'], 'dpe':metric['/dpe'], 'abcc':metric['/abcc'], 'mcdp_a':metric['/mcdp_a'], 'mcdp0':metric['/mcdp0']}


    with open(f'../results/{args.dataset}/{args.model}_{str(args.lam)}_{args.seed}.pkl', 'wb') as f:
        pickle.dump([setting_dict, val_dict, test_dict, target_hat_list, target_list, sensitive_list], f)
