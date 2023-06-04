import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv

class FirstNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(FirstNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, num_classes)
        self.dropout = 0.1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv4(x, edge_index)

        x = pyg_nn.global_max_pool(x, batch)

        return F.log_softmax(x, dim=1)


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = float(0.8)
        self.num_layers = int(10)

        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model('GAT')
        self.convs = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.batchnorm_layers.append(nn.BatchNorm1d(hidden_dim))
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
            self.batchnorm_layers.append(nn.BatchNorm1d(hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(3*hidden_dim, 3*hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(3*hidden_dim, output_dim))


    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch = batch.to(self.device)
        for i, conv in enumerate(self.convs):
            # print(x,edge_index)
            x = conv(torch.Tensor(x).to(self.device), torch.Tensor(edge_index).to(self.device))
            x = F.relu(x)
            x = self.batchnorm_layers[i](x)
            x = F.dropout(x, self.dropout, training=self.training)  # N x embedding size

        # concatenate max_pool, mean_pool and embedding of first node (i.e. the news root)
        x1 = pyg_nn.global_max_pool(x.to(self.device), batch.to(self.device)) # shape batch_size * embedding size
        x2 = pyg_nn.global_mean_pool(x.to(self.device), batch.to(self.device))

        batch_size = x1.size(0)
        indices_first_nodes = [(data.batch == i).nonzero()[0] for i in range(batch_size)]
        x3 = x[indices_first_nodes, :]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
class GATGNNModel:
    def train(self,data):
        on_gpu = True
        if on_gpu:
            print("Using gpu")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_test_split = 0.7
        test_val_split = 0.2
        train_len = int(len(data) * train_test_split)
        test_len = int(len(data) * test_val_split)  + train_len
        train_data = data[:train_len]
        test_data = data[train_len:test_len]
        val_data = data[test_len:]
        train_data_loader = torch_geometric.loader.DataLoader(train_data, batch_size=128, shuffle=True)
        val_data_loader = torch_geometric.loader.DataLoader(val_data, batch_size=128, shuffle=True)
        test_data_loader = torch_geometric.loader.DataLoader(test_data, batch_size=128, shuffle=True)
    
        print("Number of node features", 300)
        print("Dimension of hidden space", 10)
    
        model = GNNStack(300,10,2)
        if on_gpu:
            model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=5e-4)
        for epoch in range(40):
            model.train()
            epoch_loss = 0
            for batch in train_data_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out.to(device), batch.y.to(device))
                epoch_loss += loss.sum().item()
                loss.backward()
                optimizer.step()
            print("epoch", epoch, "loss:", epoch_loss / len(train_data_loader))
            if epoch%1==0:
                model.eval()
                correct = 0
                n_samples = 0
                samples_per_label = [0,0]
                pred_per_label = [0,0]
                correct_per_label = [0,0]
                with torch.no_grad():
                    for batch in train_data_loader:
                        _, pred = model(batch.to(device)).max(dim=1)
                        correct += float(pred.eq(batch.y).sum().item())
                        for i in range(2):
                            batch_i = batch.y.eq(i)
                            pred_i = pred.eq(i)
                            samples_per_label[i] += batch_i.sum().item()
                            pred_per_label[i] += pred_i.sum().item()
                            correct_per_label[i] += (batch_i*pred_i).sum().item()
                        n_samples += len(batch.y)
                train_acc = correct / n_samples
                print("Accuracy", train_acc, epoch)
                print('Training accuracy: {:.4f}'.format(train_acc))
    
                # Evaluation on the validation set 
                model.eval()
                correct = 0
                n_samples = 0
                samples_per_label = 2
                pred_per_label = 2
                correct_per_label = 2
                with torch.no_grad():
                    for batch in val_data_loader:
                        _, pred = model(batch).max(dim=1)
                        correct += float(pred.eq(batch.y.to(device)).sum().item())
                        n_samples += len(batch.y)
                val_acc = correct / n_samples
                print("Accuracy", val_acc, epoch)
                print('Validation accuracy: {:.4f}'.format(val_acc))
                
                     
                model.eval()
                correct = 0
                n_samples = 0
                with torch.no_grad():
                    for batch in test_data_loader:
                        _, pred = model(batch).max(dim=1)
                        correct += float(pred.eq(batch.y.to(device)).sum().item())
                        n_samples += len(batch.y)
                test_acc = correct / n_samples
                print("Accuracy", test_acc, epoch)
                print('Test accuracy: {:.4f}'.format(test_acc))