import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics

from dataset import PT_FEATURE_SIZE

import torch.nn.functional as F
import pennylane as qml

###############################

def GetVQC(n_qubits, qnn_layers, qnn_type):
    if qnn_type == 'ReUploadingVQC':
        def ReUploadingVQC(inputs, entangling_weights, embedding_weights):
            '''
            A variational quantum circuit (VQC) with data re-uploading
            '''
            # Prepare all zero state
            all_zero_state = torch.zeros(n_qubits)
            qml.BasisStatePreparation(all_zero_state, wires=range(n_qubits))
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for i in range(qnn_layers):
                # Variational layer
                qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
                # Encoding layer
                features = inputs * embedding_weights[i]
                qml.AngleEmbedding(features=features, wires=range(n_qubits))
            # Last varitional layer
            qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] # [qml.expval(qml.PauliZ(wires=0))]
        # Get weight shape
        entangling_weights_shape = (qnn_layers+1, ) + qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        embedding_weights_shape = (qnn_layers, n_qubits)
        weight_shapes = {
          'entangling_weights': entangling_weights_shape,
          'embedding_weights': embedding_weights_shape
        }
        return ReUploadingVQC, weight_shapes
    elif qnn_type == 'NormalVQC':
        def NormalVQC(inputs, entangling_weights):
            '''
            A variational quantum circuit (VQC) (without data re-uploading)
            '''
            qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(entangling_weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        entangling_weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=qnn_layers, n_wires=n_qubits)
        weight_shapes = {'entangling_weights': entangling_weights_shape}
        return NormalVQC, weight_shapes

qnn_type ='ReUploadingVQC' #'NormalVQC'
n_qubits = 10
qnn_layers = 20 #2
embedding_weights_shape = (qnn_layers, n_qubits)

CHAR_SMI_SET_LEN = 64

counter = 0


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d) 

    def forward(self, input):
        global counter
        output = self.conv(input)
        # print(counter ,'io shape :',input.shape, output.shape)
        counter += 1
        return output


class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        # print(f'in: {input.shape} | conv: {output1.shape}')
        output1 = self.br1(output1)
        #print("input_A_batchnorm", np.shape(output1))    
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
#
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
#
        ## merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
#
        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        #output = d1#self.br2(combine)
        #print("input_A_out", np.shape(output))
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0) # third #################################################################################
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        #print("input_B", np.shape(input))
        output1 = self.c1(input)
        # print(f'in: {input.shape} | conv: {output1.shape}')
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        #print("output_B", np.shape(output))
        return output


class DeepDTAF(nn.Module):

    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)

        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})
        #self.qlayerC = qml.qnn.TorchLayer(qnodeC, weight_shapes_updateC)
        dev = qml.device("default.qubit", wires=n_qubits)#qml.device(QPU_device = "default.qubit", wires=n_qubits, torch_device='cpu')
        VQC, weight_shapes = GetVQC(n_qubits, qnn_layers, qnn_type)
        qnode = qml.QNode(VQC, dev, interface='torch', diff_method='best')  
        self.qlayerC = qml.qnn.TorchLayer(qnode, weight_shapes)
        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:#[32, 64, 64, seq_oc]
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        # (N, H=32, L)
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            tmp = nn.Conv1d(ic, oc, 3) # fourth #################################################################################
            # print(f'ic: {seq_embed_size} | oc: {oc} | conv: {tmp}')
            conv_pkt.append(tmp)  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        
        
        self.cat_dropout = nn.Dropout(0.2)
        
        #self.classifier = nn.Sequential(
        #     nn.Linear(seq_oc+pkt_oc+smi_oc, 128),
        #    
        #     nn.Dropout(0.5),
        #     nn.PReLU(),
        #     nn.Linear(128, 64),
        #     nn.Dropout(0.5),
        #     nn.PReLU(),
        #     nn.Linear(64,1),
        #     nn.PReLU())
        
        self.clf = nn.Sequential(
            nn.Linear(10,1), # 9 7
            nn.PReLU())
        self.clangle = nn.Linear(384,10) # 9 7
        
    def forward(self, seq, pkt, smi):
        # assert seq.shape == (N,L,43)
        seq_embed = self.seq_embed(seq)  # (N,L,32)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        #qconv_seq = []
        #qic = 128
        #for oc in [128]:
        #    qconv_seq.append(self.qlayerA(seq_embed))
        #conv_seq.append(nn.AdaptiveMaxPool1d(1)) 
        #qconv_seq.append(Squeeze())
        #self.conv_seq = nn.Sequential(*qconv_seq) 
        #print("This",np.shape(seq_embed))
        #seq_embed =F.pad(seq_embed, (0, 24))  # (N, 1024, L)
        seq_conv = self.conv_seq(seq_embed)#self.qlayerA(seq_embed)
        #print("This",np.shape(seq_conv))
        #seq_conv = self.qlayerA(seq_conv)
        #seq_conv = seq_conv.squeeze(2)
        
        #seq_conv = seq_conv.squeeze()
        #seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,43)
        pkt_embed = self.seq_embed(pkt)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)
        
        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        #print(np.shape(seq_conv))
        #print(np.shape(pkt_conv))
        #print(np.shape(smi_conv))
        cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        
        # output_ = self.classifier(cat)

        #output_ = F.pad(cat, pad=(0, 512-384), mode='constant', value=0)
        output_ = self.clangle(cat)
        qout = self.qlayerC(output_)
        output = self.clf(qout)

        return output


def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation
