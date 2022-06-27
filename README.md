# DLM-DTI

The dual-language model-based drug-target interactions

Our proposed DLM-DTI simultaneously utilizes a molecule language model and a protein language model to predict binding affinity score between molecule-protein sequence pair.


![figure_2](https://user-images.githubusercontent.com/37280722/175483490-30386864-03d3-40e7-b5a1-7818dd8420d5.jpeg)


## Implementaion environment

- python 3.8
- torch 1.10
- a NVIDIA A100 GPU



## Approach

1. Pretraining
    - Molecule BERT
        - BERT architecture base masked language modeling
        - source data downloaded from PubChem database
            - using FTP, we downloaded databse
            - for more details see, https://pubchemdocs.ncbi.nlm.nih.gov/downloads
        - Main idea of molecule pretrained modeling approach was inspired by [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](http://proceedings.mlr.press/v106/shin19a/shin19a.pdf)
    - Protein BERT
        - BERT architecture base masked language modeling
        - base model was downloaded from hugging face model hub
        - https://huggingface.co/Rostlab/prot_bert
        - for more details about pretraining, please visit below urls,
            - https://github.com/agemagician/ProtTrans
            - https://www.biorxiv.org/content/10.1101/2020.07.12.199554v1
        - We fine-tuned ProtTrans BERT to our drug-target interactions database
2. Fine-tuning
    - Using pretrained molecule and protein language model, we computed binding affinity score
    - The DAVIS dataset was utilized to train and test the performance of our DLM-DTI



## Results

|              |    CI     |    MSE    | $ r_{m}^{2} $ |   AUPRC   |
| :----------: | :-------: | :-------: | :-----------: | :-------: |
|   KronRLS    |   0.871   |   0.379   |     0.407     |   0.661   |
|   SimBoost   |   0.836   |   0.282   |     0.644     |   0.709   |
|   DeepDTA    |   0.878   |   0.261   |     0.630     |   0.714   |
|   GANsDTA    |   0.880   |   0.271   |     0.653     |   0.691   |
|   DeepCDA    |   0.891   |   0.248   |     0.649     |   0.739   |
|    MT-DTI    |   0.887   |   0.245   |     0.665     |   0.730   |
| Affinity2Vec |   0.887   |   0.240   |     0.693     |   0.734   |
| **DLM-DTI**  | **0.890** | **0.193** |   **0.696**   | **0.764** |

where CI is concordance index.



## Folder hierarchy

```
project
│   Molecule_BERT_pretraining.ipynb
│   Protein_BERT_pretraining.ipynb
│   DTI_davis_512.ipynb
└── data
│   └── drug
│   │   └── molecule_tokenizer
│   └── target
│       └── protein_tokenizer
└── weights
    │   dlm_dti_davis_512
    │   molecule_bert
    │   protein_bert
```



## Source data download

[Google Drive](https://drive.google.com/drive/folders/1uSonFluSPDa6WleCO8PBbEf9d-Mhn1L5?usp=sharing)
