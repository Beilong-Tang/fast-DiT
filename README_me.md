### Steup

- remove 'name' field in environment.yml when installing

- add 
    ```shell
    export HF_HUB_CACHE=<path>/hf-cache    
    export HF_HOME=<path>/hf
    ```
    to avoid installing in home directory.

- run `sample.py` first to download the autoencoder together with pretrained DiT.
    - run `sample.py` and specify the vae to be `ema`

- (optional) run `pip install datasets` to install datasets for hf.

- the environment installs torch on cpu likely if install from login node, hence, we need to reinstall
    ```shell
    # works on ncsu hpc for h100 gpu
    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```