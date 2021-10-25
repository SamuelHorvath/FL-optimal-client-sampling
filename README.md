 ## Optimal Client Sampling for Federated Learning

 This is a Python 3 implementation of Federated Averaging (FedAvg) algorithm with [Optimal Client Sampling](https://arxiv.org/pdf/2010.13723.pdf). The code is based on [TensorFlow Federated (TFF)](https://github.com/tensorflow/federated) and is an extension of simple FedAvg example provided in [TFF examples](https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/examples/simple_fedavg). For detailed description of the method, please read our [manuscript](https://arxiv.org/pdf/2010.13723.pdf).
 
 ### Install and Test Dependencies
 Set up a new environment and install dependencies:
 ```sh
 conda create -n fl python=3.7
 conda activate fl
 pip install tensorflow_federated==0.16.1
 pip install nest_asyncio==1.4.0
 ```

 For the EMNIST experiments, unbalanced datasets modified from the [EMNIST dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FRZQIKP&version=DRAFT). They are expected to be located in [dataset](dataset) directory.

 Run the following command to test dependencies:

 ```sh
 python emnist_fedavg_main_cookup.py --total_rounds 2
 ```
 Details on the parameters can be found in the scripts.

 ### Reference
 In case you find the method or code useful for your research, please consider citing

 ```
@article{chen2020optimal,
  title={Optimal Client Sampling for Federated Learning},
  author={Chen, Wenlin and Horvath, Samuel and Richtarik, Peter},
  journal={arXiv preprint arXiv:2010.13723},
  year={2020}
}
 ```
 ### License
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

