# FedAdaDB
suplementary material for the paper: Data-Bound Adaptive Federated Learning: FedAdaDB

## Run Tuning:
  /usr/bin/python train.py -a fedAdadb -d cifar100 -t tuning -r 250 -s official -c 10 -e 1
## Run Training:
  /usr/bin/python train.py -a fedAdadb -d cifar100 -t training -r 2000 -s official -c 10 -e 4
