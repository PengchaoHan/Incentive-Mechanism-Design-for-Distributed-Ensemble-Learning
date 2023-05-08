import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_data_files')))
import collections
from functools import reduce
from models.resnet import *
from models.lenet import *
from models.wresnet import *
from torch.autograd import Variable
import numpy as np
from config import *



LOSS_ACC_BATCH_SIZE = 128   # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE
"""
class model_in_pool():
    def __init__(self, model,results_file_name=results_file_name):
        self.next_model=None
        self.own_model=model
        self.results_file_name=results_file_name
    def add_link(self, nm):
        self.next_model=nm
    def write_accuracy(self,learner_label,data_train_loader): ## write the accuracy of each learner during the test to the result.csv
        # first get the accuracy
        loss_value, train_accuracy = self.own_model.accuracy(data_train_loader, None, device)
        with open(results_file_name, 'a') as f:
            f.write(str(learner_label) + ',' + str(loss_value) + ','
                    + str(train_accuracy)+ '\n')
            f.close()
class model_pool():
    def __init__(self):
        self.size=0
        self.head=None
        self.tail=None
    def add_models(self,new):
        if self.size==0:
            self.head=new
            self.tail=new
        else:
            self.tail.add_link(new)
            self.tail=new
        self.size+=1
    def voting(self, data_test_loader,device,w=None):
        total_correct = 0
        ptr=self.head
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):## for each picture
                images, labels = Variable(images).to(device), Variable(labels).to(device)## load img
                final_decision=np.zeros((num_of_base_learners,images.shape[0]),dtype=np.int64)
                count=0
                while ptr!=None:
                    if w is not None:
                        ptr.own_model.assign_weight(w)
                    ptr.own_model.model.eval()
                    output = ptr.own_model.model(images)## predict output
                    pred = output.data.max(1)[1] # get the most possible answer
                    #ptr.write_accuracy(i,data_test_loader)
                    pred_array=pred.numpy() ## turn the tensor to a numpy array
                    final_decision[count,:]=pred_array ## stick the predition together
                    count+=1
                    ptr=ptr.next_model ## move on to next learner
            # next, get the mode
            #final_decision=np.delete(final_decision,0,axis=0) ## delete the first row
            # now the matrix size must be 10*256
            # get the matrix transpose 256*10 now
                final_decision=final_decision.T
                # the answer vector should be 256*1
                decision=np.zeros(images.shape[0],dtype=np.int64)
                row=0
                for j in final_decision: # iterate through each row
                    vals,counts = np.unique(j, return_counts=True)
                    index = np.argmax(counts)
                    decision[row]=vals[index] # get the mode for each row
                    row+=1
                result=torch.tensor(decision)
                total_correct += result.eq(labels.data.view_as(result)).sum()
                ptr=self.head
            ptr=self.head           
            count=0
            while ptr!=None:
                ptr.write_accuracy(count,data_test_loader)
                count+=1
                ptr=ptr.next_model
            acc = float(total_correct) / len(data_test_loader.dataset)
            return acc

"""


class Models():
    def __init__(self, rand_seed=None, learning_rate=0.001, num_classes=10, model_name='LeNet5', channels=1, img_size=32, device=torch.device('cuda'), flatten_weight=False, optimizer='Adam'):
        super(Models, self).__init__()
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.model = None
        self.loss_fn = None
        self.weights_key_list = None
        self.weights_size_list = None
        self.weights_num_list = None
        self.optimizer = None
        self.channels = channels
        self.img_size = img_size
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate
        self.device = device

        if model_name == 'ModelCNNMnist':
            from models.cnn_mnist import ModelCNNMnist
            self.model = ModelCNNMnist().to(device)
            self.init_variables()
        elif model_name == 'ModelCNNCifar10':
            from models.cnn_cifar10 import ModelCNNCifar10
            self.model = ModelCNNCifar10().to(device)
            self.init_variables()
        elif model_name == 'LeNet5':
            self.model = LeNet5(in_channels=channels)
        elif model_name == 'LeNet5Half':
            self.model = LeNet5Half(in_channels=channels)
        elif model_name == 'ResNet50':
            self.model = ResNet50(num_classes=num_classes)
        elif model_name == 'ResNet34':
            self.model = ResNet34(num_classes=num_classes)
        elif model_name == 'ResNet18':
            self.model = ResNet18(num_classes=num_classes)
        elif model_name == 'WResNet40-2':
            self.model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'WResNet34-2':
            self.model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
        elif model_name == 'WResNet16-1':
            self.model = WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WResNet10-2':
            self.model = WideResNet(depth=10, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'WResNet10-1':
            self.model = WideResNet(depth=10, num_classes=num_classes, widen_factor=1, dropRate=0.0)

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def weight_variable(self, tensor, mean, std):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

    def bias_variable(self, shape):
        return torch.ones(shape) * 0.1

    def init_variables(self):

        self._get_weight_info()

        weight_dic = collections.OrderedDict()

        for i in range(len(self.weights_key_list)):
            if i%2 == 0:
                tensor = torch.zeros(self.weights_size_list[i])
                sub_weight = self.weight_variable(tensor, 0, 0.1)
            else:
                sub_weight = self.bias_variable(self.weights_size_list[i])
            weight_dic[self.weights_key_list[i]] = sub_weight

        self.model.load_state_dict(weight_dic)

    def _get_weight_info(self):
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w=0
            self.weights_num_list.append(num_w)
        self.grad_key_list = []  # For the different part of weight compared to gradient
        for k, _ in self.model.named_parameters():
            self.grad_key_list.append(k)
        self.diff_index_list = []
        j = 0
        for i in range(len(self.weights_key_list)):
            if self.weights_key_list[i] == self.grad_key_list[j]:
                j += 1
            else:
                self.diff_index_list.append(i)

    def assign_weight(self, w):
        self.model.load_state_dict(w)

    def _data_reshape(self, imgs, labels=None):
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                _, y_label = torch.max(labels.data, 1)  # From one-hot to number
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    def accuracy(self, data_test_loader, w, device):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        total_correct = 0
        avg_loss = 0.0
        count=0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                count+=1
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)
        acc = float(total_correct) / len(data_test_loader.dataset)

        return avg_loss.item(), acc

    def predict(self, img, w, device):

        self.assign_weight(w)
        img, _ = self._data_reshape(img)
        with torch.no_grad():
            self.model.eval()
            _, pred = torch.max(self.model(img.to(device)).data, 1)

        return pred
