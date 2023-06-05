# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:14:42 2023

@author: bvtp1
"""
import torch
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GATConv
from torch.nn import Linear
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # Graph Convolutions
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        # Readout
        self.lin_news = Linear(in_channels, hidden_channels)
        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(2*hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Graph Convolutions
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        # Pooling
        h = gmp(h, batch)

        # Readout
        h = self.lin0(h).relu()

        # According to UPFD paper: Include raw word2vec embeddings of news 
        # This is done per graph in the batch
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        # root is e.g. [   0,   14,   94,  171,  230,  302, ... ]
        news = x[root]
        news = self.lin_news(news).relu()
        
        out = self.lin1(torch.cat([h, news], dim=-1))
        return torch.sigmoid(out)
class GCNModel:
    def __init__(self,train_loader,test_loader):
        self.train_loader = train_loader
        self.test_loader=test_loader
        self.pred = []
        self.highest_acc = 0
    def fit(self):
        train_loader=self.train_loader
        test_loader=self.test_loader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNN(300, 128, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        loss_fnc = torch.nn.BCELoss()
        def train(epoch):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                # print(out.shape,data.y.shape)
                loss = loss_fnc(torch.reshape(out, (-1,)), data.y.float())
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.num_graphs
            return total_loss / len(train_loader)
        
        @torch.no_grad()
        def test(epoch):
            model.eval()
            total_loss = 0
            all_preds = []
            all_labels = []
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = loss_fnc(torch.reshape(out, (-1,)), data.y.float())
                total_loss += float(loss) * data.num_graphs
                all_preds.append(torch.reshape(out, (-1,)).cpu())
                all_labels.append(data.y.float().cpu())
        
            # Calculate Metrics
            accuracy, f1 = metrics(all_preds, all_labels)
        
            return total_loss / len(test_loader), accuracy, f1
        
        
        def metrics(preds, gts):
            preds = torch.round(torch.cat(preds))
            gts = torch.cat(gts)
            acc = accuracy_score(preds, gts)
            f1 = f1_score(preds, gts)
            return acc, f1
        for epoch in range(40):
            train_loss = train(epoch)
            test_loss, test_acc, test_f1 = test(epoch)
            print(f'Epoch: {epoch:02d} |  TrainLoss: {train_loss:.2f} | '
                  f'TestLoss: {test_loss:.2f} | TestAcc: {test_acc:.4f} | TestF1: {test_f1:.2f}')
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.batch)
            df = pd.DataFrame()
            df["pred_logit"] = pred.cpu().detach().numpy()[:,0]
            df["pred"] = torch.round(pred).cpu().detach().numpy()[:,0]
            df["true"] = data.y.cpu().numpy()
            self.pred.append(df['pred'].values)
            
            # print(df.head(10))

        
