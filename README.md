# DeepPROTACs

This model is to predict degradation of the target given PROTACs complex.

## Training:
1. Prepare the data. This script needs the ligase pocket, target pocket, ligase ligand, target ligand, linker and the label. Here we have a script  "prepare_data.ipynb" which we use to extract ligand pocket from the complex without linker.

2. Extract features. With the ligands, pockets, linker and label of the complex, we can get the processed feature using "prepare_data.py". Here, we provide the processed "data" file of the case study in the paper. (You can use it as a toy dataset)

3. Prepare the environment. Here we export our anaconda environment as the file "env.yaml". You can use the command:
   ```shell
    conda env create -f env.yaml
    conda activate DeepPROTACs
   ```
   to get the same environment. Also, we use an RTX3090 to accelerate our 
   training.

    Besides, we highly recommond to install openbabel (2.3.2) (https://openbabel.org/wiki/Main_Page) and preprocess the mol2 files.
    ```shell
    apt install openbabel
    ```

5. Run the training script.
   ```shell
   python main.py
   ```

## Case Study:
Please run "case_study.ipynb"

## Testing:
+ Please visit https://bailab.siais.shanghaitech.edu.cn/services/deepprotacs/ 
to test. You should submit the  ligase pocket, target pocket, ligase ligand,
target ligand, and linker to get the result. 

### BaiLab