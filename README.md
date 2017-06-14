# nv-adaptive
Project exploring adaptively choosing experiments for the NV center in diamond

## Installation

I recommend using conda, with an environment set up using:

```bash
$ conda install nb_conda
$ conda env create -f environment.yml
```

### Note for Windows users:

I suggest using powershell with chocolately. You can find an installation guide here: https://chocolatey.org/install

If you don't have them, use powershell install some relevant programs with:

    choco install git putty poshgit anaconda3
    
Then `nv-adaptive` can be installed using 

    git clone git@github.com:ihincks/nv-adaptive.git
    cd nv-adaptive
    conda install nb_conda
    conda env create -f environment.yml
    
Then you can open up jupyter with

    cd src
    source activate nvmeas
    jupyter notebook
    
### Note for Linux/OSX users:

Similar to above, but you can use bash and miniconda.

