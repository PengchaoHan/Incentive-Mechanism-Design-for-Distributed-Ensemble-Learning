# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning
from torch.utils.data import DataLoader
from config import *
from datasets.dataset import *
from models.get_model import get_model, adjust_learning_rate_cifar10
from statistic.collect_stat import CollectStatisticsDEL
import numpy as np
import random
from util.utils import DatasetSplit
from util.voting import majority_voting, double_fault
import copy


random.seed(seed)
np.random.seed(seed)  # numpy
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

data_train, data_test = load_data(dataset, dataset_file_path, model_name)
data_train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=False, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=0)
img_size, channels, num_classes = get_data_info(dataset, model_name)

data_samples = [100,200,300,400,500,600,700,800,900,1000, 2000, 3000,4000, 5000,6000, 7000,8000,9000, 10000, 12000, 14000, 16000,18000, 20000]

comments = dataset + "_" + model_name + "_lr" + str(step_size) + "_n" + str(num_of_base_learners)
results_file_name_diversity = os.path.join(results_file_path, 'rst_' + comments + '_div.csv')

with open(results_file_name_diversity, 'a') as f:
    f.write("Num. samples" + ',' + "Num. iters" + ',' + 'diversity' + ',' + 'evl_accu' + ',' + 'real_accu' + '\n')
    f.close()

for d in range(len(data_samples)):

    # Create distributed dataloader
    # -------------bagging
    sample_indices = [i for i in range(len(data_train))]
    train_loader_list = []
    dataiter_list = []
    indices_list = []
    for n in range(num_of_base_learners):
        indices = random.choices(sample_indices, k=data_samples[d])  # random sample with replacement
        indices_list.append(indices)
        train_loader_list.append(DataLoader(DatasetSplit(data_train, indices), batch_size=batch_size_train, shuffle=True))
        dataiter_list.append(iter(train_loader_list[n]))

    def sample_minibatch(n):
        try:
            images, labels = next(dataiter_list[n])
            if len(images) < batch_size_train:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])
        except StopIteration:
            dataiter_list[n] = iter(train_loader_list[n])
            images, labels = next(dataiter_list[n])

        return images, labels

    results_file_name = os.path.join(results_file_path, 'rst_' + comments + '_d' + str(data_samples[d]) + '.csv')
    stat = CollectStatisticsDEL(results_file_name=results_file_name)

    breakpoints = [1000]
    # breakpoints = [30, 50, 100, 200, 500, 1000]
    model_list = [[] for i in range(len(breakpoints))]
    num_samples_bagging = data_samples[d]

    # Train models
    for n in range(num_of_base_learners):
        model = get_model(model_name, dataset, rand_seed=seed, step_size=step_size, device=device)
        # torch.save(model, os.path.join(results_file_path, 'model'))  # test model size
        num_iter = 0
        last_output = 0
        while True:
            if dataset == 'cifar10':
                adjust_learning_rate_cifar10(model.optimizer, num_iter, step_size)
            images, labels = sample_minibatch(n)
            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()
            output = model.model(images)
            loss = model.loss_fn(output, labels)
            loss.backward()
            model.optimizer.step()
            num_iter += 1

            if num_iter - last_output >= num_iter_one_output:
                stat.collect_stat_global(n, num_iter, model, train_loader_list[n], data_test_loader)
                last_output = num_iter
            for i in range(len(breakpoints)):
                if num_iter == breakpoints[i]:
                    model_list[i].append(copy.deepcopy(model.model))
                    # model_save=os.path.join(os.path.dirname(__file__)+'/model_records/'+'rst_' + comments + '_breakpoints_'+str(breakpoints[0]))+'_'+str(n)+'.pt'
                    # print(model_save)
                    # torch.save(model.model,model_save)

            if num_iter >= max_iter:
               # model_list[0].append(copy.deepcopy(model.model))
                break
    for i in range(len(breakpoints)):
        # Majority voting
        acc,accuracy_list = majority_voting(num_of_base_learners, model_list[i], data_test_loader, num_classes, device)
        # print("Ensemble test accuracy ", "Learners' accuracy list")
        # print(acc, accuracy_list)
        stat.write_voting_accuracy(acc, accuracy_list)

        # Double Fault Diversity
        df, dfa = double_fault(num_of_base_learners, model_list[i], data_train_loader, device, indices_list)  # 返回部分和全部的diversity
        # Evaluated ensemble accuracy
        avg_accu = np.mean(np.array(accuracy_list))
        evl_acc = dfa + (avg_accu - 1) / (num_of_base_learners - 1)
        with open(results_file_name_diversity, 'a') as f:
            f.write(str(data_samples[d]) + ',' + str(breakpoints[i]) + ',' + str(dfa) + ',' + str(evl_acc) + ',' + str(acc) + '\n')
            f.close()
        # print(df)


