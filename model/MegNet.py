import torch
from torch import nn
from Layers.layers import block, ff, out

class megnet(torch.nn.Module):
    def __init__(self, n1, n2, n3, nblocks, final_dim, bs, device):
        """
        Initialize the MegNet model.

        Args:
            n1 (list): List of dimensions for the first set of feed-forward layers.
            n2 (list): List of dimensions for the second set of feed-forward layers.
            n3 (list): List of dimensions for the third set of feed-forward layers.
            nblocks (int): Number of blocks.
            final_dim (int): Dimension of the final output.
            bs (int): Batch size.
            device (torch.device): Device for tensor computations.
        """
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.device = device
        self.nblocks = nblocks
        
        self.ff_list = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        dims = [n1, n2, n3]
        
        for i in range(nblocks):
            self.ff_list.append(ff(dims[0][i]))
            self.ff_list.append(ff(dims[1][i]))
            self.ff_list.append(ff(dims[2][i]))
            
        for i in range(nblocks):
            for j in range(len(dims[0][i])):
                self.blocks.append(block(self.ff_list, i, dims[0][i][j][1], dims[1][i][j][1], dims[2][i][j][1], bs, device))
            
        self.output_scc = out((2 * final_dim)+4, 1)
    def forward(self, data, combinations, edge_data):
        """
        Forward pass through the MegNet model.

        Args:
            data (dict): Dictionary containing input data including node features, edge features, global features,
                         atom4bond, and bond4atom.
            combinations (tensor): Tensor containing combinations.

        Returns:
            tuple: A tuple containing the model's outputs.
        """
        node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom = data['node_ftr'], data['edge_ftr'], torch.zeros((1, 4)).float(), data['atom4bond'], data['bond4atom']
        
        node_ftr = torch.tensor(node_ftr).to(self.device).float()
        edge_ftr = torch.tensor(edge_ftr).to(self.device).float()
        atom4bond = torch.tensor(atom4bond).to(self.device).float()
        bond4atom = torch.tensor(bond4atom).to(self.device).float()
        gbl_ftr = gbl_ftr.to(self.device)
        
        num = node_ftr.shape[1]
        items = list(range(num))
        
        node_ftr = self.ff_list[0](node_ftr)
        edge_ftr = self.ff_list[1](edge_ftr)
        gbl_ftr = self.ff_list[2](gbl_ftr)
        
        for i in range(self.nblocks):
            if i >= 1:
                node_ftr = self.ff_list[3 * i](node_ftr)
                edge_ftr = self.ff_list[3 * i + 1](edge_ftr)
                gbl_ftr = self.ff_list[3 * i + 2](gbl_ftr)
                
            node_ftr_1, edge_ftr_1, gbl_ftr_1 = self.blocks[i](node_ftr, edge_ftr, gbl_ftr, atom4bond, bond4atom, i)
            
            node_ftr += node_ftr_1
            edge_ftr += edge_ftr_1
            gbl_ftr += gbl_ftr_1
        ftr = node_ftr[0, combinations.int(), :]
        ftr = ftr.flatten(2)
        ftr = torch.cat([ftr,edge_data],-1)
        return (
            torch.squeeze(self.output_scc(ftr), -1)
        )
