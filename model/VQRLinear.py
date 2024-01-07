from pennylane import numpy as np
import torch
from torch import nn
import pennylane as qml


class VQRLinearModel(nn.Module):

    def __init__(self, n_qubit: int, layers: int = 1, duplicate_qubits: bool = False) -> None:
        super().__init__()
        self.n_qubit = n_qubit * 2 if duplicate_qubits else n_qubit
        print(f'Init the SimpleVQR with {self.n_qubit} qubits')
        # initialize thetas (or weights) of NN
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=self.n_qubit)
        initial_weights = np.pi * np.random.random(shape, requires_grad=True)
        self.weights = nn.Parameter(torch.from_numpy(initial_weights), requires_grad=True)
        # initialize bias of NN
        self.bias = nn.Parameter(torch.from_numpy(np.zeros(1)), requires_grad=True)

    def encoder(self, x):
        cnt = 0
        for f in x:
            qml.RX(np.arctan(f) * 2, wires=cnt)
            cnt += 1

    def layer(self, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubit))

    def circuit(self, x):
        self.encoder(x)
        qml.Barrier(wires=range(self.n_qubit), only_visual=True)
        self.layer(self.weights)
        qml.Barrier(wires=range(self.n_qubit), only_visual=True)
        return qml.expval(qml.PauliZ(wires=0))

    def forward(self, X):
        # define the characteristics of the device
        dev = qml.device("default.qubit", wires=self.n_qubit)
        vqc = qml.QNode(self.circuit, dev, interface="torch")
        res = []
        for x in X:
            res.append(vqc(x) + self.bias)
        res = torch.stack(res)
        return res

    def draw_circuit(self, style: str = 'pennylane'):
        qml.draw_mpl(self.circuit, decimals=2, style=style, wire_order=range(self.n_qubit))(
            [x.item() for x in np.random.random(self.n_qubit, requires_grad=False)])
