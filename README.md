# Incentive-Mechanism-Design-for-Distributed-Ensemble-Learning
Incentive Mechanism Design for Distributed Ensemble Learning


#### To run experiments:

The code runs on Python 3. To run distributed ensemble learning (bagging) with 100 base learners, run
```
python simulation_bagging.py -n 100 
```


To fit the ensemble accuracy, run
```
python fitting_mnist_3d.py 
```


To conduct incentive mechanism, run
```
python simulation_DEL_incentive.py 
```