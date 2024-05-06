import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tabulate import tabulate

import pickle

from dataset import load_adult_data, load_bank_marketing_data
from utils import seed_everything, PandasDataSet, print_metrics, clear_lines, InfiniteDataLoader
from metrics import metric_evaluation
from networks import MLP

from loss import Mixup



def train_step(model, data, target, sensitive, scheduler, optimizer, clf_criterion, fair_criterion, lam, device, args=None):
    (X0,X1), (y0,y1), (s0,s1) = data, target, sensitive
    model.train()
    optimizer.zero_grad()

    _, output0 = model(X0)
    _, output1 = model(X1)
    clf_loss = clf_criterion(torch.cat([output0,output1]), torch.cat([y0,y1]))
    fair_loss = fair_criterion(X0, X1, model)

    loss = clf_loss + lam * fair_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model



def test(model, test_loader, clf_criterion, fair_criterion, lam, device, prefix="test", args=None):
    model.eval()
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
            h, output = model(data)

            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)

    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}", epsilon_list=args.epsilon_list, K=args.K)
    return metric



def InfiniteDataLoader_v2(dataset, batch_size):
    while True:
        for _ in range(1000):
            indices = torch.from_numpy(np.random.choice(len(dataset),batch_size))
            yield dataset[indices]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/adult/raw", help="Path to the dataset folder")
    parser.add_argument("--dataset", type=str, default="adult", help="Choose a dataset from the available options: adult, bank_marketing")
    parser.add_argument("--model", type=str, default="mixup", help="Model type")
    parser.add_argument("--target_attr", type=str, default="income", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="sex", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--evaluation_metrics", type=str, default="ap,dp,abcc,mcdp0", help="Evaluation metrics separated by commas")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency")

    parser.add_argument("--lam", type=float, default=0.1)

    parser.add_argument("--num_training_steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=1024, help="Batch size for evaluation")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mlp_layers", type=str, default="512,256,64", help="MLP layers as comma-separated values, e.g., 512,256,64")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="uuid", help="Experiment name")
    parser.add_argument("--save", type=int, default=0, help="Saving the well-trained model locally")
    parser.add_argument("--epsilon", type=float, default=0.01, help="epsilon for computing mcdp")
    parser.add_argument("--K", type=int, default=1, help="K for computing mcdp")

    args = parser.parse_args()
    args.epsilon_list = [0.001,0.01,0.02,0.05,0.1] # you may manually specify epsilon here

    table = tabulate([(k, v) for k, v in vars(args).items()], tablefmt='grid')
    print(table)


    seed_everything(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = ".."
    if args.dataset == "adult":
        print(f"Dataset: adult")
        X, y, s = load_adult_data(path=root_dir+"/datasets/adult/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "bank_marketing":
        print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path=root_dir+"/datasets/bank_marketing/raw", sensitive_attribute=args.sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)


    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=args.seed)
    X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(X_testvalid, y_testvalid, s_testvalid, test_size=0.5, stratify=y_testvalid, random_state=args.seed)

    dataset_stats = {
        "dataset": args.dataset,
        "num_features": X.shape[1],
        "num_classes": len(np.unique(y)),
        "num_sensitive": len(np.unique(s)),
        "num_samples": X.shape[0],
        "num_train": X_train.shape[0],
        "num_val": X_val.shape[0],
        "num_test": X_test.shape[0],
        "num_y1": (y.values == 1).sum(),
        "num_y0": (y.values == 0).sum(),
        "num_s1": (s.values == 1).sum(),
        "num_s0": (s.values == 0).sum(),
    }


    # Create the table using the tabulate function
    table = tabulate([(k, v) for k, v in dataset_stats.items()], tablefmt='grid')

    print(table)


    numurical_cols = X.select_dtypes("float32").columns
    if len(numurical_cols) > 0:
        # scaler = StandardScaler().fit(X[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        scaler = StandardScaler().fit(X_train[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        X_train[numurical_cols] = X_train[numurical_cols].pipe(scale_df, scaler)
        X_val[numurical_cols]   = X_val[numurical_cols].pipe(scale_df, scaler)
        X_test[numurical_cols]  = X_test[numurical_cols].pipe(scale_df, scaler)

    idx0, idx1 = (s_train.values==0).reshape(-1), (s_train.values==1).reshape(-1)
    X0, y0, s0 = X_train[idx0], y_train[idx0], s_train[idx0]
    X1, y1, s1 = X_train[idx1], y_train[idx1], s_train[idx1]

    train_data = PandasDataSet(X_train, y_train, s_train)
    train_data_0, train_data_1 = PandasDataSet(X0, y0, s0), PandasDataSet(X1, y1, s1)
    val_data, test_data = PandasDataSet(X_val, y_val, s_val), PandasDataSet(X_test, y_test, s_test)

    if args.dataset == 'bank_marketing':
        train_infinite_loader_0 = InfiniteDataLoader_v2(train_data_0, batch_size=int(args.batch_size/2))
    else:
        train_infinite_loader_0 = InfiniteDataLoader(train_data_0, batch_size=int(args.batch_size/2), shuffle=True, drop_last=True)
    train_infinite_loader_1 = InfiniteDataLoader(train_data_1, batch_size=int(args.batch_size/2), shuffle=True, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.evaluation_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.evaluation_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.evaluation_batch_size, shuffle=False)



    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]
    net = MLP(n_features=n_features, num_classes=1, mlp_layers=mlp_layers ).to(device)
    clf_criterion = nn.BCELoss()
    fair_criterion = Mixup()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    print(net)


    logs = []
    headers = ["Step(Tr|Val|Te)"] + args.evaluation_metrics.split(",")


    for step, ((X0, y0, s0), (X1, y1, s1)) in enumerate(zip(train_infinite_loader_0, train_infinite_loader_1)):
        if step >= args.num_training_steps:
            break
        X0, y0, s0, X1, y1, s1 = X0.to(device), y0.to(device), s0.to(device), X1.to(device), y1.to(device), s1.to(device)
        net = train_step(model=net, data=(X0,X1), target=(y0,y1), sensitive=(s0,s1), optimizer=optimizer, scheduler=scheduler,  clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam,  device=device)

        if step % args.log_freq == 0 or step == 1 or step == args.num_training_steps:

            train_metrics = test(model=net, test_loader=train_loader, clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="train", args=args)
            val_metrics   = test(model=net, test_loader=val_loader,   clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="val", args=args)
            test_metrics  = test(model=net, test_loader=test_loader,  clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="test", args=args)
            res_dict = {}
            res_dict.update(train_metrics)
            res_dict.update(val_metrics)
            res_dict.update(test_metrics)

            # for printing
            if step % (args.log_freq*10) == 0:
                res = print_metrics(res_dict, args.evaluation_metrics, train=True)
                logs.append( [ step, *res] )
                if  step > 3:
                    clear_lines(len(logs)*2 + 1)
                table = tabulate(logs, headers=headers, tablefmt="grid", floatfmt="02.2f")
                print(table)


    with torch.no_grad():   # Final eval/test
        setting_dict = {'dataset':args.dataset, 'method':args.model, 'sensitive_attr':args.sensitive_attr, 'target_attr':args.target_attr, 'lam':args.lam, 'seed':args.seed}
        net.eval()

        target_hat_list, target_list, sensitive_list = [], [], []
        for data, target, sensitive in val_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
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
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
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
