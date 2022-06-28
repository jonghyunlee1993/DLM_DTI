import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from multiprocessing import Pool


def check_qed_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            qed_score = QED.qed(mol)
            if qed_score >= 0.7:
                return True
            else:
                return False
        except Exception as e:
            return False
        
def pool_filter(pool, func, smiles):
    return [s for s, keep in zip(smiles, pool.imap(func, tqdm(smiles))) if keep]



if __name__ == "__main__":
    with open("chem_total.pkl", "rb") as f:
        data = pickle.load(f)

    entire_smiles = list(data.values())
    
    with Pool() as p:
        filtered_smiles = pool_filter(p, check_qed_score, entire_smiles)
        
    with open("chem_qed_filtered.txt", "w") as f:
        f.write("\n".join(filtered_smiles))
