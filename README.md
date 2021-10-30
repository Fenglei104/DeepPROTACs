# DeepPROTACs

This model is to predict degradation of the target given PROTACs complex.

## Training:
1. Unzip the "DeepPROTACs_code.tgz" and come into the directory
   ```shell
   tar zxvf DeepPROTACs_code
   cd DeepPROTACS_code
   ```
2. Prepare the data. This script needs the ligase pocket, target pocket, ligase 
    ligand, target ligand, linker and the label. Here we have a script 
    "prepare_data.ipynb" which we use to extract the pocket from the complex, 
    and encoding them. From the script, you can get 11 pkl file, which are 
    inputs of the network.

3. Prepare the environment. Here we export our anaconda environment as the
   file "env.yaml". You can use the command:
   ```shell
    conda env create -f env.yaml
    conda activate DeepPROTACs
   ```
   to get the same environment. Also, we use an RTX3090 to accelerate our 
   training.

4. Run the training script.
   ```shell
   python main.py
   ```

## Testing:
+ Please visit https://bailab.siais.shanghaitech.edu.cn/services/deepprotacs/ 
to test. You should submit the  ligase pocket, target pocket, ligase ligand,
target ligand, and linker to get the result. 

### BaiLab