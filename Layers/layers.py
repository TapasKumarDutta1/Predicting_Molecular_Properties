import torch
from torch import nn

class SoftplusLayer(nn.Module):
    """
    Custom Softplus layer implementation.

    This layer computes the Softplus activation function.

    Softplus activation function: f(x) = log(1 + exp(x))

    Args:
        None

    Inputs:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the Softplus activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x) + torch.log(0.5 * torch.exp(-torch.abs(x)) + 0.5)

class ff(nn.Module):
    """
    Feedforward Neural Network model.

    This model consists of a sequence of linear layers followed by Softplus activation.

    Args:
        hiddens (list of tuples): List of tuples where each tuple contains the number of input and output features for a linear layer.

    Inputs:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after passing through the model.
    """
    def __init__(self, hiddens):
        super().__init__()
        layers = nn.ModuleList([])
        for i, o in hiddens:
            layers.append(nn.Linear(in_features=i, out_features=o))
            layers.append(SoftplusLayer())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class out(nn.Module):
    """
    Output Neural Network model.

    This model is designed for the final output layer of a neural network.

    Args:
        in_ftr (int): Number of input features.
        out_ftr (int): Number of output features.

    Inputs:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after passing through the model.
    """
    def __init__(self, in_ftr, out_ftr):
        super().__init__()
        layers = nn.ModuleList([])
        layers.append(nn.Linear(in_features=in_ftr, out_features=in_ftr // 2))
        layers.append(SoftplusLayer())
        layers.append(nn.Linear(in_features=in_ftr // 2, out_features=in_ftr // 4))
        layers.append(SoftplusLayer())
        layers.append(nn.Linear(in_features=in_ftr // 4, out_features=out_ftr))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class megnet_layer(nn.Module):
    def __init__(self, bond_hidden, atom_hidden, unv_hidden, bs, device):
        """
        Initialize the MegNet Layer.

        Args:
            bond_hidden (int): Hidden dimension for bond features.
            atom_hidden (int): Hidden dimension for atom features.
            unv_hidden (int): Hidden dimension for global features.
            bs (int): Batch size.
            device (torch.device): Device for tensor computations.
        """
        super().__init__()
        self.bs = torch.tensor(list(range(bs)))
        self.bond_mlp = ff([[bond_hidden * 4, bond_hidden]])
        self.atom_mlp = ff([[atom_hidden * 3, atom_hidden]])
        self.unv_mlp = ff([[unv_hidden * 3, unv_hidden]])
        self.blank = torch.zeros(1).to(device)
        
    def update_bond(self, inputs):
        """
        Update bond features.

        Args:
            inputs (list): List of input tensors including node features, edge features, global features,
                           atom4bond, and bond4atom.

        Returns:
            torch.Tensor: Updated bond features.
        """
        node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom = inputs
        a = node_ftr[self.bs, atom4bond.int(), :]  # 4,2,d
        a = a.flatten(2)  # 4, 2d
        b, n, d = edge_ftr.shape
        cat = torch.cat([a, edge_ftr, torch.unsqueeze(gbl_ftr, 1).repeat(1, n, 1)], -1)  # 1, 4, 2d; 1,4,d; 1,4,d
        return self.bond_mlp(cat)  # 1,4,d
        
    def aggeregate_bond(self, e_p, inputs):
        """
        Aggregate bond features.

        Args:
            e_p (torch.Tensor): Bond features.
            inputs (list): List of input tensors including node features, edge features, global features,
                           atom4bond, and bond4atom.

        Returns:
            torch.Tensor: Aggregated bond features.
        """
        node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom = inputs
        _, _, c = e_p.shape
        blank = self.blank.repeat(c)
        blank = blank.view(1, 1, c)
        e_p = torch.cat([e_p, blank], 1)  # 1,5,d
        a = e_p[0, bond4atom.int(), :]  # 1,5,4,d
        summation = a.sum(dim=2)  # 1,5,d
        mask = bond4atom > -1
        mask = torch.unsqueeze(mask.sum(-1), -1)
        out = summation / torch.max(torch.ones_like(mask), mask)
        return out
        
    def update_atom(self, b_ei_p, inputs):
        """
        Update atom features.

        Args:
            b_ei_p (torch.Tensor): Updated bond features.
            inputs (list): List of input tensors including node features, edge features, global features,
                           atom4bond, and bond4atom.

        Returns:
            torch.Tensor: Updated atom features.
        """
        node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom = inputs
        _, n, _ = node_ftr.shape
        cat = torch.cat([b_ei_p, node_ftr, torch.unsqueeze(gbl_ftr, 1).repeat(1, n, 1)], -1)
        return self.atom_mlp(cat)
        
    def update_global(self, e_p, v_p, inputs):
        """
        Update global features.

        Args:
            e_p (torch.Tensor): Bond features.
            v_p (torch.Tensor): Updated atom features.
            inputs (list): List of input tensors including node features, edge features, global features,
                           atom4bond, and bond4atom.

        Returns:
            torch.Tensor: Updated global features.
        """
        node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom = inputs
        
        cat = torch.cat([torch.mean(e_p, 1), torch.mean(v_p, 1), gbl_ftr], -1)
        return self.unv_mlp(cat)
    
    def forward(self, inputs):
        """
        Forward pass through the MegNet Layer.

        Args:
            inputs (list): List of input tensors including node features, edge features, global features,
                           atom4bond, and bond4atom.

        Returns:
            list: A list containing updated 'v_p', 'e_p', and 'u_p'.
        """
        e_p = self.update_bond(inputs)
        b_ei_p = self.aggeregate_bond(e_p, inputs)
        v_p = self.update_atom(b_ei_p, inputs)
        u_p = self.update_global(e_p, v_p, inputs)
        return [v_p, e_p, u_p]
    
class block(nn.Module):
    def __init__(self, ff_list, n, bond_hidden, atom_hidden, unv_hidden, bs, device):
        """
        Initialize the block in the MegNet model.

        Args:
            ff_list (nn.ModuleList): List of FeedForwardNN layers.
            n (int): Block index.
            bond_hidden (int): Hidden dimension for bond features.
            atom_hidden (int): Hidden dimension for atom features.
            unv_hidden (int): Hidden dimension for global features.
            bs (int): Batch size.
            device (torch.device): Device for tensor computations.
        """
        super().__init__()
        self.ff_list = ff_list
        self.layer = megnet_layer(bond_hidden, atom_hidden, unv_hidden, bs, device)
        self.n = n
        self.drop_node = nn.Dropout()
        self.drop_edge = nn.Dropout()
        self.drop_global = nn.Dropout()
        
    def forward(self, node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom, n):
        """
        Forward pass through a block in the MegNet model.

        Args:
            node_ftr (torch.Tensor): Node features.
            edge_ftr (torch.Tensor): Edge features.
            gbl_ftr (torch.Tensor): Global features.
            atom4bond (torch.Tensor): Atom-to-bond mapping.
            bond4atom (torch.Tensor): Bond-to-atom mapping.
            n (int): Block index.

        Returns:
            tuple: A tuple containing updated 'node_ftr', 'edge_ftr', and 'gbl_ftr'.
        """
        out = self.layer([node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom])
        
        node_ftr, edge_ftr, gbl_ftr = out
        
        node_ftr = self.drop_node(node_ftr)
        edge_ftr = self.drop_edge(edge_ftr)
        gbl_ftr = self.drop_global(gbl_ftr)
        
        return node_ftr, edge_ftr, gbl_ftr
