# NeuCall


### Environment
Tested on Ubuntu 20.04 with
 - Anaconda
 - Angr
 - CUDA 11.8
 - ./env/angr_environment.yml
 - ./env/model_environment.yml

### Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/project-name.git
   cd project-name

2. Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#)
3. Generate graph
   1. Create a conda env

        ```bash
        conda env create -f ./env/angr_environment.yml
        conda activate py3.6

        ```
   2. Download [binary] and [groundtruth]

        ```bash
        python3 angrcfg.py
        ```
4. Run neucall
   1. Create a new conda env
        ```bash
        conda env create -f ./env/model_environment.yml
        conda activate py3.8
        ```
    2. Run model
        ```bash
        python3 neucall.py
        ```
