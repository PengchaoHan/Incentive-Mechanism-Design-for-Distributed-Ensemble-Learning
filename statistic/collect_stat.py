import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv'):
        self.results_file_name = results_file_name
        #self.results_file_name = os.path.dirname(__file__)+'/results.csv'
        print(self.results_file_name)
        with open(self.results_file_name, 'a') as f:
            f.write(
                'num_iter,loss_value,train_Accuracy\n')
            f.close()

    def collect_stat_global(self, num_iter, model, train_data_loader, test_data_loader, w_global=None):
        loss_value, train_accuracy = model.accuracy(train_data_loader, w_global, device)
        _, prediction_accuracy = model.accuracy(test_data_loader, w_global, device)

        print("Iter. " + str(num_iter) + "  train accu " + str(train_accuracy) + "  test accu " + str(prediction_accuracy))

        with open(self.results_file_name, 'a') as f:
            f.write(str(num_iter) + ',' + str(loss_value) + ','
                    + str(train_accuracy) + ',' + str(prediction_accuracy) + '\n')
            f.close()


class CollectStatisticsDEL:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv'):
        self.results_file_name = results_file_name
        #self.results_file_name = os.path.dirname(__file__)+'/results.csv'
        print(self.results_file_name)
        with open(self.results_file_name, 'a') as f:
            f.write(
                'learner_id, num_iter, loss_value,train_Accuracy,test_Accuracy\n')
            f.close()

    def collect_stat_global(self, learner_id, num_iter, model, train_data_loader, test_data_loader, w_global=None):
        loss_value, train_accuracy = model.accuracy(train_data_loader, w_global, device)
        _, prediction_accuracy = model.accuracy(test_data_loader, w_global, device)

        print("Learner " + str(learner_id) + " Iter. " + str(num_iter) + "  train accu " + str(train_accuracy) + "  test accu " + str(prediction_accuracy))

        with open(self.results_file_name, 'a') as f:
            f.write(str(learner_id) + ',' + str(num_iter) + ',' + str(loss_value) + ','
                    + str(train_accuracy) + ',' + str(prediction_accuracy) + '\n')
            f.close()

    def write_voting_accuracy(self, acc,accuracy_list): ## write the accuracy after voting, add a ----------to separate the result of each iteration
        with open(self.results_file_name, 'a') as f:
            # f.write("Ensemble test accuracy" + "Learners' accuracy list" + '\n')
            f.write(str(acc) + ',' + str(accuracy_list)+'\n')
            f.close()

