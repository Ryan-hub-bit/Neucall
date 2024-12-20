import argparse
import time

import torch as th
import dgl, os
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
# from torchmetrics.classification import BinaryRecall
from torch import tensor

import database, random

import dgl.nn as dglnn

device = None

# node is the output of palmtree and other features(vector)
# one of GN relationational for same node and same link
# eg {"codefunc_edge: n * 128 + 2 + pe -> 128"}
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Define the first layer of heterogeneous graph convolution
        self.conv1 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(in_feats[rel[0]], hid_feats)
            for rel in rel_names}, aggregate='sum')

        # Define the second layer of heterogeneous graph convolution
        self.conv2 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')

        # Define the third layer of heterogeneous graph convolution
        self.conv3 = dglnn.HeteroGraphConv({
            rel[1]: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # Apply the first convolution layer
        h = self.conv1(graph, inputs)
        # Apply ReLU activation and dropout to each node type's features
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        # Apply the second convolution layer
        h = self.conv2(graph, h)
        # Apply ReLU activation and dropout to each node type's features
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        # Apply the third convolution layer
        h = self.conv3(graph, h)

        return h

class LinkPredictor(th.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        # Initialize a ModuleList to hold the linear layers
        self.lins = th.nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.lins.append(th.nn.Linear(in_channels, hidden_channels))

        # Middle layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.lins.append(th.nn.Linear(hidden_channels, hidden_channels))

        # Last layer: hidden_channels -> out_channels
        self.lins.append(th.nn.Linear(hidden_channels, out_channels))

        # Dropout rate
        self.dropout = dropout

        # Initialize the parameters of the linear layers
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters for all linear layers
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        # Pass input through each linear layer except the last one
        for lin in self.lins[:-1]:
            x = lin(x)           # Apply linear transformation
            x = F.relu(x)        # Apply ReLU activation
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Apply the final linear transformation
        x = self.lins[-1](x)

        # Apply sigmoid activation to get a probability score
        return th.sigmoid(x)


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, dropout):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names, dropout)

    def forward(self, g, x):
        h = self.sage(g, x)
        return h

def init_dataset(Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe = False):
    dataset = database.load(Revedges=Revedges, Calledges=Calledges, Laplacian_pe=Laplacian_pe,
                 Adddata = Adddata, Addfunc = Addfunc, DataRefedgs = DataRefedgs, CodeRefedgs = CodeRefedgs)

    # rev -> reverse
    rel_names = [('code', 'code2func_edges', 'func'),
                 ('code', 'code2code_edges', 'code'),
                 ('code', 'codecall_edges', 'code'),
                 ('code', 'codexrefcode_edges', 'code'),
                 ('code', 'codexrefdata_edges', 'data'),
                 ('data', 'dataxrefcode_edges', 'code'),
                 ('data', 'dataxrefdata_edges', 'data'),
                 ('code', 'rev_code2code_edges', 'code'),
                 ('code', 'rev_codecall_edges', 'code'),
                 ('code', 'rev_codexrefcode_edges', 'code'),
                 ('data', 'rev_codexrefdata_edges', 'code'),
                 ('code', 'rev_dataxrefcode_edges', 'data'),
                 ('data', 'rev_dataxrefdata_edges', 'data'),
                 ('func', 'rev_code2func_edges', 'code')
                 ]
    if not Revedges:
        rel_names = [o for o in rel_names if not o[1].startswith('rev_')]
    if not Adddata:
        rel_names = [o for o in rel_names if o[0]!='data'and o[2]!='data']
    if not Addfunc:
        rel_names = [o for o in rel_names if o[0]!='func'and o[2]!='func']
    if not DataRefedgs:
        rel_names = [o for o in rel_names if not o[1].endswith('xrefdata_edges')]
    if not CodeRefedgs:
        rel_names = [o for o in rel_names if not o[1].endswith('xrefcode_edges')]
    if not Calledges:
        rel_names = [o for o in rel_names if not o[1].endswith('codecall_edges')]

    return dataset, rel_names


# database.load
def get_one_graph(dataset, i, Adddata = True, Addfunc = True, Laplacian_pe=False):
    print(f"graph{i} is loading")
    g, glabels = dataset[i]
    g = g.to(device)
    if Adddata:
        node_features = {
            'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float(),
            'data': g.nodes['data'].data['feat'].view(g.nodes['data'].data['feat'].shape[0], -1).float()}
    else:
        node_features = {
            'code': g.nodes['code'].data['feat'].view(g.nodes['code'].data['feat'].shape[0], -1).float()}
    if Addfunc:
        if Laplacian_pe:
            node_features['func'] = g.nodes['func'].data['feat'].view(g.nodes['func'].data['feat'].shape[0], -1).float()
        else:
            node_features['func'] = th.zeros(g.num_nodes("func")).view(-1,1).float().to(device)

    return g, glabels, node_features

# exp_all  different setting
def exp_all(epochs = 6, hidden_features = 128, inslen = 0, savePATH = '/home/isec/model/result/all', Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe=False):
    if not os.path.exists(savePATH):
        os.makedirs(savePATH)
    dataset, rel_names = init_dataset(Revedges=Revedges, Adddata=Adddata, Addfunc=Addfunc, DataRefedgs = DataRefedgs, Calledges = Calledges, CodeRefedgs = CodeRefedgs, Laplacian_pe=Laplacian_pe)
    pe = 0
    if Laplacian_pe:
        pe = 2
    # +2 float basicblock_addr + function_addr
    in_features = {'code':inslen*128+2+pe, 'data': 1+pe, 'func': 1+int(pe/2)}
    if not Adddata:
        in_features.pop('data')
    if not Addfunc:
        in_features.pop('func')
    # GN and linkpredictor model
    model = Model(in_features, hidden_features, hidden_features, rel_names, dropout = 0.2)
    predictor = LinkPredictor(hidden_features*2, hidden_features, 1, 3, 0)
    model, predictor = map(lambda x: x.to(device), (model, predictor))
    opt = th.optim.Adam(model.parameters())
    model.float()
    predictor.float()
    bestf1 = 0
    f1 = []
    precision = []
    recall = []
    auroc = []
    if os.path.exists(os.path.join(savePATH, 'predictor.checkpoint')):
        model.load_state_dict(th.load(os.path.join(savePATH, 'model.checkpoint')))
        predictor.load_state_dict(th.load(os.path.join(savePATH, 'predictor.checkpoint')))
        with open(os.path.join(savePATH, 'bestf1.txt'), 'r') as f:
            bestf1 = float(f.read())
        with open(os.path.join(savePATH, 'f1s.txt'), 'r') as f:
            slist = f.read().split('\n')
            slist.remove('')
            for i in slist:
                i = i.split(' ')
                f1.append(float(i[0]))
                precision.append(float(i[1]))
                recall.append(float(i[2]))
                auroc.append(float(i[3]))
    model.train()
    predictor.train()
    timep = []
    timep.append(time.time())

    trains = [0, int(dataset.__len__()*0.8)]
    valids = [int(dataset.__len__()*0.8), int(dataset.__len__()*0.9)]
    tests = [int(dataset.__len__()*0.9), dataset.__len__()]

    metric = torchmetrics.F1Score(task="binary")
    num = 0
    #smallPE = True
    if Laplacian_pe:
        randomlist = []
        for i in range(dataset.__len__()):
            if i in skips:
                continue
            graphfile = os.path.join(dataset.directory, str(i) + '.graphpe')
            if os.path.exists(graphfile) or os.path.getsize(graphfile[:-2])<70000000:
                randomlist.append(i)
        tmp = randomlist.__len__()
        #print(f'data num = {tmp}')
        validlist = randomlist[int(tmp*0.9):tmp]
        randomlist = randomlist[:int(tmp*0.9)]
    else:
        randomlist = list(range(trains[1]))
        for i in skips:
            if i in randomlist:
                randomlist.remove(i)
        validlist = list(range(valids[0], valids[1]))

    for epoch in range(epochs):
        random.shuffle(randomlist)
        #print(randomlist)
        for i in range(len(randomlist)):
            num += 1
            g, glabels, node_features = get_one_graph(dataset=dataset, i=randomlist[i], Adddata = Adddata, Addfunc = Addfunc, Laplacian_pe=Laplacian_pe)
            pred = model(g, node_features)
            edge = glabels['GT_edges']
            pos_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
            pos_loss = -th.log(pos_out + 1e-15).mean()
            edge = glabels['GT_F_edges']
            neg_out = predictor(th.cat((pred['code'][edge[0]],pred['code'][edge[1]]),dim=1))#(pred['code'][edge[0]], pred['code'][edge[1]])
            neg_loss = -th.log(1 - neg_out + 1e-15).mean()
            loss = pos_loss + neg_loss
            opt.zero_grad()
            # backward
            loss.backward()
            opt.step()
            timep.append(time.time())
            pos_loss = None
            neg_loss = None
            loss = None
            g = None
            edge = None
            pos_out = None
            neg_out = None
            neg_loss = None
            loss = None

            if num%50 == 0:
                preds = []
                targets = []
                model.eval()
                predictor.eval()
                for i in validlist:
                    if i in skips:
                        continue
                    g, glabels, node_features = get_one_graph(dataset=dataset, i=i, Adddata = Adddata, Addfunc = Addfunc, Laplacian_pe=Laplacian_pe)
                    pred = model(g, node_features)
                    edge = glabels['GT_edges']
                    pos_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                               dim=1))
                    edge = glabels['GT_F_edges']
                    neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                               dim=1))
                    preds+=pos_out.tolist()
                    preds+=neg_out.tolist()
                    targets+=[[1]]*pos_out.shape[0]
                    targets+=[[0]]*neg_out.shape[0]
                f1.append(metric(th.tensor(preds), th.tensor(targets)).item())
                precision_tensor = torchmetrics.functional.precision(th.tensor(preds), th.tensor(targets),"binary")
                recall_tensor = torchmetrics.functional.recall(th.tensor(preds), th.tensor(targets),"binary")
                auroc.append(torchmetrics.functional.auroc(th.tensor(preds), th.tensor(targets),"binary").item())
                precision.append(precision_tensor.item())
                recall.append(recall_tensor.item())
                if bestf1 <f1[-1]:
                    bestf1 = f1[-1]
                    th.save(model.state_dict(), os.path.join(savePATH, 'model.checkpoint'))
                    th.save(predictor.state_dict(), os.path.join(savePATH, 'predictor.checkpoint'))
                    with open(os.path.join(savePATH, 'bestf1.txt'), 'w') as f:
                        f.write(str(bestf1))
                with open(os.path.join(savePATH, 'f1s.txt'), 'w') as f:
                    for i in range(len(f1)):
                        f.write(str(f1[i])+' '+str(precision[i])+' '+str(recall[i])+' '+str(auroc[i])+' '+'\n')
                model.train()
                predictor.train()
    preds = []
    targets = []
    model.eval()
    predictor.eval()
    for i in tests:
        g, glabels, node_features = get_one_graph(dataset=dataset, i=i, Adddata = Adddata, Addfunc = Addfunc, Laplacian_pe=Laplacian_pe)
        pred = model(g, node_features)
        edge = glabels['GT_edges']
        pos_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                    dim=1))
        edge = glabels['GT_F_edges']
        neg_out = predictor(th.cat((pred['code'][edge[0]], pred['code'][edge[1]]),
                                    dim=1))
        preds+=pos_out.tolist()
        preds+=neg_out.tolist()
        targets+=[[1]]*pos_out.shape[0]
        targets+=[[0]]*neg_out.shape[0]
        f1 = metric(th.tensor(preds), th.tensor(targets)).item()
        precision_tensor = torchmetrics.functional.precision(th.tensor(preds), th.tensor(targets),"binary")
        recall_tensor = torchmetrics.functional.recall(th.tensor(preds), th.tensor(targets),"binary")
        auroc = torchmetrics.functional.auroc(th.tensor(preds), th.tensor(targets),"binary").item()
        precision = precision_tensor.item()
        recall = recall_tensor.item()
        print(f"f1 :{f1}, precision :{precision}, recall:{recall}, auroc: {auroc}")

if __name__ == "__main__":
# def main():
    epochs = 10
    skips = []
    hidden_features = 512
    inslen = 70
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    savedir = "/home/isec/test/model" + "/result" +"_" + str(inslen)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen,  savePATH = savedir + '/allpe',
            Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe=True)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen,  savePATH = savedir + '/alloff',
            Revedges = False, Adddata = False, Addfunc = False, DataRefedgs = False, Calledges = False, CodeRefedgs = False, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen,savePATH = savedir + '/revedges',
            Revedges = True, Adddata = False, Addfunc = False, DataRefedgs = False, Calledges = False, CodeRefedgs = False, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen=inslen, savePATH = savedir + '/adddata',
            Revedges = False, Adddata = True, Addfunc = False, DataRefedgs = True, Calledges = False, CodeRefedgs = False, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen, savePATH = savedir + '/addfunc',
            Revedges = False, Adddata = False, Addfunc = True, DataRefedgs = False, Calledges = False, CodeRefedgs = False, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features,inslen = inslen, savePATH = savedir + '/calledges',
            Revedges = False, Adddata = False, Addfunc = False, DataRefedgs = False, Calledges = True, CodeRefedgs = False, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen, savePATH = savedir + '/codeRefedgs',
            Revedges = False, Adddata = False, Addfunc = False, DataRefedgs = False, Calledges = False, CodeRefedgs = True, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen, savePATH = savedir + '/allafter',
            Revedges = True, Adddata = True, Addfunc = True, DataRefedgs = True, Calledges = True, CodeRefedgs = True, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen,  savePATH = savedir + '/dataedgecoderef',
            Revedges = False, Adddata = True, Addfunc = False, DataRefedgs = True, Calledges = False, CodeRefedgs = True, Laplacian_pe=False)
    exp_all(epochs = epochs, hidden_features=hidden_features, inslen = inslen,  savePATH = savedir + '/pe',
            Revedges = False, Adddata = False, Addfunc = False, DataRefedgs = False, Calledges = False, CodeRefedgs = False, Laplacian_pe=True)