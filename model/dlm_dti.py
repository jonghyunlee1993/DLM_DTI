import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class DualLanguageModelDTI(nn.Module):
    def __init__(self, 
                 molecule_encoder, protein_encoder, hidden_dim=512,
                 molecule_input_dim=128, protein_input_dim=1024):
        super().__init__()
        self.molecule_encoder = molecule_encoder
        self.protein_encoder = protein_encoder
        
        # model freezing without last layer
        for param in self.molecule_encoder.encoder.layer[0:-1].parameters():
            param.requires_grad = False        
        for param in self.protein_encoder.encoder.layer[0:-1].parameters():
            param.requires_grad = False
        
        self.molecule_align = nn.Sequential(
            nn.LayerNorm(molecule_input_dim),
            nn.Linear(molecule_input_dim, hidden_dim)
        )
        
        self.protein_align = nn.Sequential(
            nn.LayerNorm(protein_input_dim),
            nn.Linear(protein_input_dim, hidden_dim)
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, molecule_seq, protein_seq):
        encoded_molecule = self.molecule_encoder(**molecule_seq)
        encoded_protein = self.protein_encoder(**protein_seq)
        
        cls_molecule = encoded_molecule.pooler_output
        cls_protein = encoded_protein.pooler_output
        
        cls_molecule = self.molecule_align(cls_molecule)
        cls_protein = self.protein_align(cls_protein)
        
        cls_concat = torch.cat([cls_molecule, cls_protein], dim=1)

        x = F.gelu(self.fc1(cls_concat))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        out = self.fc_out(x)
        
        return out


def load_pretrained():
    molecule_bert = BertModel.from_pretrained("weights/molecule_bert")
    protein_bert = BertModel.from_pretrained("weights/protein_bert")

    return molecule_bert, protein_bert

