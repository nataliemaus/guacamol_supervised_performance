import torch 
import numpy as np 
import gpytorch 
from gpytorch.mlls import PredictiveLogLikelihood 
import pandas as pd 
import math 
import os 
from utils.load_data import load_molecule_train_data
from model.update_model import update_model
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt 
from utils.set_seed import set_seed 
from model.ppgpr import GPModelDKL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    task_id="rano",
    seed=0,
    n_inducing_points=1024,
):
    # set seed 
    set_seed(seed=seed)
    # Grab init data 
    all_z, all_y = load_molecule_train_data(
        task_id=task_id,
    )
    # shuffle data 
    indices = np.arange(len(all_y))
    indices = np.random.permutation(indices)
    all_y = all_y[indices] 
    all_z = all_z[indices] 
    # split train/test 
    n_test = int(len(all_y)*0.2)
    test_z = all_z[0:n_test]
    train_z = all_z[n_test:]
    test_y = all_y[0:n_test]
    train_y = all_y[n_test:]
    # define GP Model w/ DKL
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    n_pts = min(train_z.shape[0], n_inducing_points)
    model = GPModelDKL(train_z[:n_pts, :].to(device), likelihood=likelihood ).to(device)
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    model = model.eval() 
    # train model 
    model = update_model(
        model=model,
        mll=mll,
        train_z=train_z,
        train_y=train_y,
    )
    model.to(device)
    # test model 
    model.eval()
    output = model(test_z.to(device))
    test_mll = mll(output, test_y.to(device))
    mean_preds = output.mean 
    errors = mean_preds - test_y.to(device)
    mse = (errors**2).mean()
    rmse = mse**(0.5)
    test_mll = round(test_mll.item(), 2)
    rmse = round(rmse.item(), 2)
    mse = round(mse.item(), 2)
    # plot test results 
    if not os.path.exists("plots/"):
        os.mkdir("plots/")
    plt.scatter(test_y.tolist(), mean_preds.tolist())
    title_str = f"{task_id} GP w/ DKL Test RMSE:{rmse}, MLL: {test_mll}"
    plot_filename = f"plots/{task_id}_test_scatter.png"
    plt.title(title_str)
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.clf() 
    # get train performance 
    output = model(train_z.to(device))
    train_mll = mll(output, train_y.to(device))
    mean_preds = output.mean 
    errors = mean_preds - train_y.to(device)
    mse = (errors**2).mean()
    rmse = mse**(0.5)
    train_mll = round(train_mll.item(), 2)
    rmse = round(rmse.item(), 2)
    mse = round(mse.item(), 2)
    # plot train results 
    plt.scatter(train_y.tolist(), mean_preds.tolist())
    title_str = f"{task_id} GP w/ DKL Train RMSE:{rmse}, MLL: {train_mll}"
    plot_filename = f"plots/{task_id}_train_scatter.png"
    plt.title(title_str)
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.clf() 


if __name__ == "__main__":
    for task_id in ["zale","rano","siga","pdop","osmb","valt","adip","logp","dhop","shop","med1","med2"]:
        main(task_id=task_id,)
        plt.clf()


# python3 main.py 
