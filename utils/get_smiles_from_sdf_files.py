import gzip
import glob
import pickle
from tqdm import tqdm

chem = {}

flist = glob.glob("*.sdf.gz")

for index, file in enumerate(flist):
    print(f"working file number: {index + 1}\tremaining: {len(flist) - index}")
    with gzip.open(file, 'r') as f:
        data = f.readlines()
        for i, line in tqdm(enumerate(data), total=len(data)):
            if line == b'> <PUBCHEM_IUPAC_NAME>\n':
                name = data[i + 1].decode('utf-8').rstrip()
            elif line == b'> <PUBCHEM_OPENEYE_CAN_SMILES>\n':
                smile = data[i + 1].decode('utf-8').rstrip()
                chem[name] = smile

    print(f"number of processed chemicals: {len(chem)}\n")
    
    if index + 1 % 10 == 0:
        with open('chem_backup_' + str(index) + '.pkl', 'wb') as f:
            pickle.dump(chem, f)
    
with open('chem_total.pkl', 'wb') as f:
    pickle.dump(chem, f)