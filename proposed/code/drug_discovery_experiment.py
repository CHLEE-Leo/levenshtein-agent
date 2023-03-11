# %%
# SMIELS 그림 그리기
from rdkit import Chem
from rdkit.Chem import Draw

# ms_smis = [["C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])Cl", "cdnb"],
#            ["C1=CC(=CC(=C1)N)C(=O)N", "3aminobenzamide"],
#            [sample_dat['Compound_smiles'][0], 'blahblah'] ]
# ms_smis = [[ms_smis[i][0].upper(), 'blah'] for i in range(len(ms_smis))]
# ms_smis = list(outputs_decoded_hit)[:3]
# ms_smis = list(gens_decoded_hit[-20:-10])
condition1 = input('draw outputs SMILES (o) or generated SMILES (g) ?')
if condition1 == 'o':
    condition2 = input('draw all SMIELS (a) or specific (s) ?')

    if condition2 == 'a':
        ms_smis = list(outputs_decoded_hit)    

        ms_smis = [[ms_smis[i].upper(), 'blah'] for i in range(len(ms_smis))]
        ms = [[Chem.MolFromSmiles(x[0]), x[1]] for x in ms_smis]

        for i in range(len(ms)):
            if str(ms[i][0]) == 'None':
                pass
            else:
                target_idx = copy.deepcopy(i)
                display(ms[i][0])
                print('target_idx is : {}'.format(target_idx))
                print('SMILES is {}'.format(ms_smis[i]))

    elif condition2 == 's':
        target_idx = input('input specific index')
        # ms_smis = [list(outputs_decoded_hit)[-18]]
        # ms_smis = [list(gens_decoded_hit)[-18]]
        # ms_smis = [list(gens_decoded_hit)[target_idx]]
        ms_smis = [list(outputs_decoded_hit)[int(target_idx)]]
        ms_smis = [[ms_smis[i].upper(), 'blah'] for i in range(len(ms_smis))]
        ms = [[Chem.MolFromSmiles(x[0]), x[1]] for x in ms_smis]

        for i in range(len(ms)):
            display(ms[i][0])
            # Draw.MolToImage(ms[i][0], size = (400, 200))
            Draw.MolToFile(ms[i][0], '/home/messy92/Leo/NAS_folder/ICML23/molecule(base)_' + str(target_idx) + '.png', size = (400, 200))

elif condition1 == 'g':
    condition2 = input('draw all SMIELS (a) or specific (s) ?')

    if condition2 == 'a':
        ms_smis = list(gens_decoded_hit)    
        ms_smis = [[ms_smis[i].upper(), 'blah'] for i in range(len(ms_smis))]
        ms = [[Chem.MolFromSmiles(x[0]), x[1]] for x in ms_smis]

        for i in range(len(ms)):
            if str(ms[i][0]) == 'None':
                pass
            else:
                target_idx = copy.deepcopy(i)
                display(ms[i][0])
                print('target_idx is : {}'.format(target_idx))
                print('SMILES is {}'.format(ms_smis[i]))

    elif condition2 == 's':
        target_idx = input('input specific index')
        # ms_smis = [list(outputs_decoded_hit)[-18]]
        # ms_smis = [list(gens_decoded_hit)[-18]]
        # ms_smis = [list(gens_decoded_hit)[target_idx]]
        ms_smis = [list(gens_decoded_hit)[int(target_idx)]]
        ms_smis = [[ms_smis[i].upper(), 'blah'] for i in range(len(ms_smis))]
        ms = [[Chem.MolFromSmiles(x[0]), x[1]] for x in ms_smis]
        for i in range(len(ms)):
            display(ms[i][0])
            # Draw.MolToImage(ms[i][0], size = (400, 200))
            Draw.MolToFile(ms[i][0], '/home/messy92/Leo/NAS_folder/ICML23/molecule(gen)_' + str(target_idx) + '.png', size = (400, 200))


# %%
