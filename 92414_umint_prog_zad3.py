#final verzia
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import torchvision.transforms as T
import os
from PIL import Image
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#-init--------------------------------------
df = pd.read_csv('navig.csv', sep=',')
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth', 30)
do_r18 = 0
prep_misc, do_univar, do_bivar = 0,0,0
epochs, lr, patience = 100, 0.001, 4
#-constants---------------------------
PNI_DICT = dict(zip(df['PictureName'], df['Irradiance']))
IMAGES_FOLDER = 'ins\\in_small_png'
CATEG_X_COLS = ['Month', 'Hour']

#-functions and classes----------------------
def remap():
    global df
    df['DateTime'] = df['DateTime'].str.split('.').str[0]
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M:%S')
    df['Year']  = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Hour']  = df['DateTime'].dt.hour
    df.drop(columns=['DateTime'], inplace=True)
    print('(MISC/UNIVAR/BIVAR ONLY) Remapped: DateTime->Year,Month,Hour')
def dummify():
    global df
    df = pd.get_dummies(df, columns=[col for col in CATEG_X_COLS if col in df.columns])
    print('(BIVAR ONLY) Performed dummification:',CATEG_X_COLS,'dummified')

class IrradianceImageDataset():
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor()  
        ])

        self.image_names = [
            fname for fname in PNI_DICT.keys()
            if os.path.isfile(os.path.join(IMAGES_FOLDER, fname))
        ]
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(IMAGES_FOLDER, img_name)

        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        target = float(PNI_DICT[img_name])
        target = torch.tensor([target], dtype=torch.float32)

        return image, target

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56 -> 28
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
class ResNet18Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        regressor = models.resnet18(weights=weights)

        for param in regressor.parameters():
            param.requires_grad = False
        for param in regressor.layer4.parameters():
            param.requires_grad = True
        for param in regressor.fc.parameters():
            param.requires_grad = True

        in_features = regressor.fc.in_features
        regressor.fc = nn.Linear(in_features, 1)
        self.regressor = regressor
    def forward(self, x):
        return self.regressor(x)
    
def compute_regression_metrics(y_true, y_pred):
    mse  = np.mean((y_true - y_pred) ** 2)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return mse, mae, rmse, r2
def errvis(y_train, y_test, y_train_pred, y_test_pred):
    train_mse, _, _, train_r2 = compute_regression_metrics(y_train, y_train_pred)
    test_mse, _, _, test_r2 = compute_regression_metrics(y_test, y_test_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12,5))
    axes[0][0].scatter(y_train_pred, y_train-y_train_pred, alpha=0.6, color='red')
    axes[0][0].axhline(0, color='black', linestyle='--')
    axes[0][0].set_title(f'Train Residuals, MSE={train_mse:.4f}, R2={train_r2:.4f}')
    axes[0][0].set_xlabel('Predicted')
    axes[0][0].set_ylabel('Residuals')
    axes[1][0].scatter(y_test_pred, y_test-y_test_pred, alpha=0.6, color='red')
    axes[1][0].axhline(0, color='black', linestyle='--')
    axes[1][0].set_title(f'Test Residuals, MSE={test_mse:.4f}, R2={test_r2:.4f}')
    axes[1][0].set_xlabel('Predicted')
    axes[1][0].set_ylabel('Residuals')
    axes[0][1].scatter(y_train_pred, y_train, alpha=0.6, color='green')
    axes[0][1].set_title(f'Train Predicted vs True')
    axes[0][1].set_xlabel('Predicted')
    axes[0][1].set_ylabel('True')
    axes[1][1].scatter(y_test_pred, y_test, alpha=0.6, color='green')
    axes[1][1].set_title(f'Test Predicted vs True')
    axes[1][1].set_xlabel('Predicted')
    axes[1][1].set_ylabel('True')
    plt.subplots_adjust(left=0.04,right=0.99,bottom=0.05,top=0.97)
    win = tk.Toplevel()
    win.title('errvis')
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text='Close', command=win.destroy).pack(pady=10)
    win.geometry('800x600')   
def priebeh(y_a, y_b, capa):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(1, len(y_a) + 1), y_a, label=f'Train {capa} (epoch)')
    ax.plot(range(1, len(y_b) + 1), y_b, label=f'Val {capa} (epoch)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(capa)
    ax.set_title(f'Train vs Val {capa} (CNN)')
    ax.legend()
    fig.tight_layout()
    win = tk.Toplevel()
    win.title(f'Train vs Val {capa} (CNN)')
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text='Close', command=win.destroy).pack(pady=10)
    win.geometry('800x600')
    
def train(model, train_loader, val_loader, device): #TODO check later?
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    model.to(device)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        epoch_stats = {}

        for phase, loader, is_train in [
            ('train', train_loader, True),
            ('val',   val_loader,   False)
            ]:
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds, targets = [], []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * xb.size(0)
                preds.append(logits.detach().cpu())
                targets.append(yb.detach().cpu())

            epoch_loss = running_loss / len(loader.dataset)
            y_pred_t = torch.cat(preds)
            y_true_t = torch.cat(targets)
            mse, mae, rmse, r2 = compute_regression_metrics(y_true_t.numpy().ravel(), y_pred_t.numpy().ravel())
            epoch_stats[phase] = {
                'loss': epoch_loss,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

        epoch_train_stats = epoch_stats['train']
        epoch_val_stats = epoch_stats['val']
        print(
            f'Epoch {epoch+1}/{epochs}:\n'
            f'Train MSE: {epoch_train_stats['mse']:.4f}, MAE: {epoch_train_stats['mae']:.4f}, RMSE: {epoch_train_stats['rmse']:.4f}, R2: {epoch_train_stats['r2']:.4f}\n'
            f'Val   MSE: {epoch_val_stats['mse']:.4f}, MAE: {epoch_val_stats['mae']:.4f}, RMSE: {epoch_val_stats['rmse']:.4f}, R2: {epoch_val_stats['r2']:.4f}'
        )
        train_losses.append(epoch_train_stats['loss'])
        val_losses.append(epoch_val_stats['loss'])
        
        if epoch_val_stats['loss'] < best_val_loss:
            best_val_loss = epoch_val_stats['loss']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            print(f'No improvement for {wait} epoch(s).')
            if wait >= patience:
                print(f'Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}')
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses
    
def prir(img_path, model, device):
    model.eval()
    with torch.no_grad():
        image = Image.open(img_path).convert('RGB')
        transform = T.Compose([
            T.ToTensor()  
        ])
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        pred = model(image)
        return pred.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = IrradianceImageDataset()
    rat_tn, rat_vd = 0.7, 0.15
    n_full = len(dataset)
    n_tn = int(rat_tn * n_full)
    n_vd   = int(rat_vd * n_full)
    n_tt  = n_full - n_tn - n_vd
    print('Split plan:',n_full,'->', [n_tn, n_vd, n_tt])
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_tn, n_vd, n_tt],
        generator=generator
    )

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if not do_r18:
        print('Regressor choice: CNNRegressor')
        model = CNNRegressor()
    else:
        print('Regressor choice: ResNet18Regressor')
        model = ResNet18Regressor()

    model, train_losses, val_losses = train(model, train_loader, val_loader, device)
    
    model.eval()
    y_train_pred, y_train_true = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            y_train_pred.append(logits.cpu())
            y_train_true.append(yb.cpu())
    y_train_pred_t = torch.cat(y_train_pred)
    y_train_true_t = torch.cat(y_train_true)

    y_test_pred, y_test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            y_test_pred.append(logits.cpu())
            y_test_true.append(yb.cpu())
    y_test_pred_t = torch.cat(y_test_pred)
    y_test_true_t = torch.cat(y_test_true)

    train_mse, train_mae, train_rmse, train_r2 = compute_regression_metrics(y_train_true_t.numpy().ravel(), y_train_pred_t.numpy().ravel())
    test_mse, test_mae, test_rmse, test_r2 = compute_regression_metrics(y_test_true_t.numpy().ravel(), y_test_pred_t.numpy().ravel())
    print(
        f'Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}\n'
        f'Test  MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}'
    )

    print('Sanity check:')
    print(prir('ins\\in_small_png\\Alpnach_20180101_11-00-00_ExposureStack_Image_10_image.png', model, device),'/ 348.24')
    print(prir('ins\\in_small_png\\Alpnach_20180101_10-45-00_ExposureStack_Image_10_image.png', model, device),'/ 136.41')
    print(prir('ins\\in_small_png\\Alpnach_20180101_15-45-00_ExposureStack_Image_10_image.png', model, device),'/ 2.57')
    print('Own pics:')
    '''print(prir('ins\\in_own_png\\20251204_080000465.JPG', model, device),'/ BA 8h')
    print(prir('ins\\in_own_png\\20251204_110054501.JPG', model, device),'/ BA 11h')
    print(prir('ins\\in_own_png\\20251204_170030174.JPG', model, device),'/ BA 17h')
    print(prir('ins\\in_own_png\\20251204_200640866.JPG', model, device),'/ BA 20h')
    print(prir('ins\\in_own_png\\20251204_230014408.JPG', model, device),'/ BA 23h')
    print(prir('ins\\in_own_png\\20251205_020424642.JPG', model, device),'/ BA 2h')
    print(prir('ins\\in_own_png\\20251204_075611.jpg', model, device),'/ TO 8h')
    print(prir('ins\\in_own_png\\20251204_110046.jpg', model, device),'/ TO 11h')
    print(prir('ins\\in_own_png\\20251204_135948.jpg', model, device),'/ TO 14h')
    print(prir('ins\\in_own_png\\20251204_170216.jpg', model, device),'/ TO 17h')
    print(prir('ins\\in_own_png\\20251204_194359.jpg', model, device),'/ TO 20h')
    print(prir('ins\\in_own_png\\20251204_215339.jpg', model, device),'/ TO 22h')'''
    
    errvis(y_train_true_t.numpy().ravel(), y_test_true_t.numpy().ravel(),
           y_train_pred_t.numpy().ravel(), y_test_pred_t.numpy().ravel())
    priebeh(train_losses, val_losses, 'Loss')
    print('Press <RETURN> to exit')
    input()
if not (prep_misc or do_univar or do_bivar):
    main()
print('OK')


if prep_misc:
    remap()
    
    q='IrradianceNotCompensated'
    x = df[q].to_numpy()
    y = df["Irradiance"].to_numpy()
    plt.scatter(x, y, s=1)
    plt.xlabel(q)
    plt.ylabel("Irradiance")
    plt.title(q+' vs Irradiance')
    plt.show()
if do_univar:
    remap()

    axes=df.hist(bins=20)
    for ax in axes.ravel():
        ax.grid(False)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.subplots_adjust(left=0.03,right=.97,top=.97,bottom=0.03)
    plt.show()
if do_bivar:
    remap()
    dummify()
    df.drop(columns=['PictureName'], inplace=True)
    
    corr=pd.concat([df.drop(columns=['Irradiance']),df['Irradiance']],axis=1).corr()
    ax = sns.heatmap(corr,annot=True,fmt='.1f',cmap='coolwarm',cbar=False,
                     xticklabels=corr.columns,yticklabels=corr.columns)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.subplots_adjust(left=0.11,right=1,top=1,bottom=0.21)
    plt.show()