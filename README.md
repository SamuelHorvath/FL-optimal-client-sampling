 ## Guidelines

 This is a python 3 implementation of optimal sampling for Federated Averaging (FedAvg) algorithm. The code is based on [TensorFlow Federated (TFF)](https://github.com/tensorflow/federated) and it is an extension of simple FedAvg example provide in [TFF examples](https://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/examples/simple_fedavg). For detailed description, please see our [manuscript](https://arxiv.org/pdf/2010.13723.pdf).
 
 ### Instal Dependencies and Run
 To install dependencies 
 ```sh
 conda create -n fl python=3.7
 conda activate fl
 pip install tensorflow_federated==0.16.1
 pip install nest_asyncio==1.4.0
 ```

 Unbalanced datasets based on [EMNIST dataset](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data) that are used for the experiments can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FRZQIKP&version=DRAFT). They are expected to be located in [dataset](dataset) directory.

 Try to run following command to test dependencies

 ```sh
 python emnist_fedavg_main_cookup.py --total_rounds 2
 ```
 Details on the parameters can be found in [emnist_fedavg_main_cookup.py](emnist_fedavg_main_cookup.py).

 ### Citing
 In case you find this code useful, please consider citing

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

