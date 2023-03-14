# DeepPROTACs

This model is to predict degradation of the target given PROTACs complex. This repository can help reproduce the experiments of:

Li, F., Hu, Q., Zhang, X. et al. DeepPROTACs is a deep learning-based targeted degradation predictor for PROTACs. Nat Commun 13, 7133 (2022). https://doi.org/10.1038/s41467-022-34807-3

## Training:
1. Prepare the data. This script needs the ligase pocket, target pocket, ligase ligand, target ligand, linker and the label. Here we have a script  "prepare_data.ipynb" which we use to extract ligand pocket from the complex without linker. If you want to use the provided examples, please modify the file path in the .ipynb file.

2. Extract features. With the ligands, pockets, linker and label of the complex, we can get the processed feature using "prepare_data.py". Here, we provide the processed "data" directory of the case study in the paper. (You can use it as a toy dataset). If you want to prepare for your own dataset with the script, please remove the "data" directory first.

3. Prepare the environment. Here we export our anaconda environment as the file "env.yaml". You can use the command:
   ```shell
    conda env create -f env.yaml
    conda activate DeepPROTACs
   ```
   to get the same environment. Also, we use an RTX3090 to accelerate our 
   training and test on Ubuntu 18.04.

    Besides, we highly recommond to install openbabel (2.3.2) (https://openbabel.org/wiki/Main_Page) and preprocess the mol2 files.
    ```shell
    apt install openbabel
    ```

5. Run the training script.
   ```shell
   python main.py
   ```

## Case Study:
Please run "case_study.ipynb". The file is to test on the toy data which contains 16 complex in the paper, and it will output the true labels and predicted labels of each complex.

## Single Prediction:
If you want just to predict one single PROTACs complex, please prepare your files into one dirctory, such as `single_test`, and name the files as `ligase_ligand.mol2`, `ligase_pocket.mol2`, `target_ligand.mol2`, `target_pocket.mol2` and `linker.smi` respectively. Then run the  `single_prediction.py` plus your dir name, like, 
```
python single_prediction.py single_test
```
The script will output one single value (0 or 1).

## Testing:
+ Please visit https://bailab.siais.shanghaitech.edu.cn/services/deepprotacs/ 
to test. You should submit the  ligase pocket, target pocket, ligase ligand,
target ligand, and linker to get the result. 

### BaiLab

* This project is free for research teams. Please do not charge for commercial purposes.
