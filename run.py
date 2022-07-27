import argparse

import torch
import pytorch_lightning as pl
from tdc.multi_pred import DTI
from pytorch_lightning.callbacks import ModelCheckpoint

from model.dlm_dti import *
from model.lightning_trainer import *
from utils.data import *
from utils.evaluation_metrics import *


class MyError(Exception):
    def __str__(self):
        return "Choose between [davis / kiba]"

class MyCollator(object):
    def __init__(self, molecule_tokenizer, protein_tokenizer):
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer


    def __call__(self, batch):
        molecule_seq, protein_seq, y = [], [], []
        
        for (molecule_seq_, protein_seq_, y_) in batch:
            molecule_seq.append(molecule_seq_)
            protein_seq.append(protein_seq_)
            y.append(y_)
            
        molecule_seq = self.molecule_tokenizer.pad(molecule_seq, return_tensors="pt")
        protein_seq = self.protein_tokenizer.pad(protein_seq, return_tensors="pt")
        y = torch.tensor(y).float()

        return molecule_seq, protein_seq, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="davis", help="davis or kiba")
    parser.add_argument("--n_train_samples", default=0, type=int, help="# of train samples (default full)")
    parser.add_argument("--n_valid_samples", default=0, type=int, help="# of valid samples (default full)")
    parser.add_argument("--batch_size", default=32, type=int, help="# of batch for training")
    parser.add_argument("--protein_max_len", default=512, type=int, help="max length of protein sequences")
    parser.add_argument("--project_name", default="DLM_DTI_default", help="wieght files will be saved in the weights/project_name folder")
    parser.add_argument("--continue_training", default="n", help="use pretrained weight file [n] / y")
    parser.add_argument("--continue_training_weight", default="", help="specify pretrained weight file")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="set learning rate for training")
    parser.add_argument("--epochs", default=100, type=int, help="set max training epochs for training")
    parser.add_argument("--n_gpu", default=0, type=int, help="select gpu number for training")
    parser.add_argument("--use_scheduler", default="n", help="use cosine annealing learning rate scheduler [n] / y")
    parser.add_argument("--use_amp", default="n", help="use automatic mixed precision [n] / y")

    args = parser.parse_args()

    if args.dataset not in ['davis', 'kiba']:
        raise MyError()

    data = DTI(name=args.dataset)
    if args.dataset == "davis":
        data.convert_to_log(form="binding")
    data_split = data.get_split()

    train_df, valid_df, test_df = data_split['train'], data_split['valid'], data_split['test']
    molecule_tokenizer, protein_tokenizer = load_tokenizer()

    train_dataset, valid_dataset, test_dataset = genereate_datasets(
        train_df, valid_df, test_df, 
        molecule_tokenizer, protein_tokenizer
    )

    collate_batch = MyCollator(molecule_tokenizer, protein_tokenizer)

    if args.n_train_samples > 0:
        train_sampler = torch.utils.data.RandomSampler(train_df, replacement=True, num_samples=args.n_train_samples)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=14, 
            pin_memory=True, prefetch_factor=10, drop_last=True, 
            collate_fn=collate_batch, sampler=train_sampler 
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=14, 
            pin_memory=True, prefetch_factor=10, drop_last=True, 
            collate_fn=collate_batch, shuffle=True 
        )
    
    if args.n_train_samples > 0:
        valid_sampler = torch.utils.data.RandomSampler(valid_df, replacement=True, num_samples=3000)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, num_workers=14, 
            pin_memory=True, prefetch_factor=10, collate_fn=collate_batch,
            sampler=valid_sampler
        )
    else:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, num_workers=14, 
            pin_memory=True, prefetch_factor=10, collate_fn=collate_batch
        )

    
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=14, 
        pin_memory=True, prefetch_factor=10, collate_fn=collate_batch
    )

    molecule_bert, protein_bert = load_pretrained()
    dlm_dti = DualLanguageModelDTI(molecule_bert, protein_bert)

    callbacks = [
        ModelCheckpoint(monitor='valid_loss', save_top_k=3, dirpath='weights/' + args.project_name, filename='dlm_dti-{epoch:03d}-{valid_loss:.4f}-{valid_mae:.4f}'),
    ]

    model = DTI_prediction(dlm_dti, lr=args.learning_rate, use_scheduler=args.use_scheduler)
    if args.continue_training == "y":
        ckpt_fname = args.continue_training_weight
        print(f"load pretrained weight file {ckpt_fname}")
        model = model.load_from_checkpoint(ckpt_fname, dlm_dti=dlm_dti)
        

    if args.use_amp == "y":
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=[args.n_gpu], enable_progress_bar=True, callbacks=callbacks, precision=16)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=[args.n_gpu], enable_progress_bar=True, callbacks=callbacks)
    trainer.fit(model, train_dataloader, valid_dataloader)
