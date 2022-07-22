import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


def load_tokenizer():
    molecule_tokenizer = molecule_tokenizer = BertTokenizer.from_pretrained("data/drug/molecule_tokenizer", model_max_length=128)
    protein_tokenizer = BertTokenizer.from_pretrained("data/target/protein_tokenizer", do_lower_case=False)

    return molecule_tokenizer, protein_tokenizer


def genereate_datasets(train_df, valid_df, test_df, molecule_tokenizer, protein_tokenizer):
    train_dataset = DTIDataset(train_df, molecule_tokenizer, protein_tokenizer)
    valid_dataset = DTIDataset(valid_df, molecule_tokenizer, protein_tokenizer)
    test_dataset = DTIDataset(test_df, molecule_tokenizer, protein_tokenizer)

    return train_dataset, valid_dataset, test_dataset


class DTIDataset(Dataset):
    def __init__(self, data, molecule_tokenizer, protein_tokenizer, protein_max_len=512):
        self.data = data
        
        self.molecule_max_len = 100
        self.protein_max_len = protein_max_len
        
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer
    
        
    def molecule_encode(self, molecule_sequence):
        molecule_sequence = self.molecule_tokenizer(
            " ".join(molecule_sequence), 
            max_length=self.molecule_max_len, 
            truncation=True
        )
        
        return molecule_sequence
    
    
    def protein_encode(self, protein_sequence):
        protein_sequence = self.protein_tokenizer(
            " ".join(protein_sequence), 
            max_length=self.protein_max_len, 
            truncation=True
        )
        
        return protein_sequence
        
        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        molecule_sequence = self.molecule_encode(self.data.loc[idx, "Drug"])
        protein_sequence = self.protein_encode(self.data.loc[idx, "Target"])
        y = self.data.loc[idx, "Y"]
                
        return molecule_sequence, protein_sequence, y