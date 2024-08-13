import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
import numpy as np
from config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import random
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1').colors

random.seed(seed)
np.random.seed(seed)  # numpy

cost_mean = 0.0001
cost_var = 0.00001
gamma_list = [100*i for i in range(5, 60)]


cost = np.array([0.00001+0.00001*i for i in range(num_of_base_learners)])

D_max = np.array([60000 for i in range(num_of_base_learners)])

min_n = 3
min_d = 200

def func_div(data, a,b,c,d,e,f,g,h,i):
    x,y = data  # x: N, y: D
    return (a*np.power(y,b)+c)*(d-e*np.log(x*f+g) + h) +i

def func_accu(data, a,b,c,d,e,f):
    return a*np.log(data*b+c)/np.log(data*d+e)+f

def func_el_accu_real(data, a,b,c,d,e,f,g,h,i):
    x,y = data
    return (a*np.log(x*b+c)+d)*(e*np.log(y*f+g) + h) +i

def func_avg_accu(data):
    n, d = data  # x: N, y: D
    avg_data = np.sum(d)/n
    diversity = func_div((n, avg_data), 1.7608163972129824e-09, 1.5104552112861804, -6.525417189616139e-05, -2522.661603100801,
                    0.097812298684595, 0.593434375934323, -1.1861713921636314, 2522.9784618367407, -1.4699885825297731e-06)
    avg_accu = 0.0
    for i in range(int(n)):
        avg_accu += func_accu(d[i], 5.7952987719351795e-06, 0.00013345338910324258, -0.010678528614487103, 2.945455527694917e-07,
                                 1.000181238936544, 0.9734099561091535)
    avg_accu /= n
    return avg_accu

def func_el_accu(data):
    n, d = data  # x: N, y: D
    avg_data = np.sum(d)/n
    diversity = func_div((n, avg_data), 1.7608163972129824e-09, 1.5104552112861804, -6.525417189616139e-05, -2522.661603100801,
                    0.097812298684595, 0.593434375934323, -1.1861713921636314, 2522.9784618367407, -1.4699885825297731e-06)
    avg_accu = 0.0
    for i in range(int(n)):
        avg_accu += func_accu(d[i], 5.7952987719351795e-06, 0.00013345338910324258, -0.010678528614487103, 2.945455527694917e-07,
                                 1.000181238936544, 0.9734099561091535)
    avg_accu /= n
    ensemble_accuracy_array = diversity + (avg_accu - 1) / (n - 1) + 1  # +1 to make it positive

    return ensemble_accuracy_array

def func_el_accu_real_(data):
    n, d = data  # x: N, y: D
    d = np.mean(d)
    ensemble_accuracy_array = func_el_accu_real((n,d), 0.00030625506538928043, 2.997047029160774, -5.866838062424441,
                                                9.29565543353712, 0.002342501188356276, 4.663997199764269,
                                                -352.23699101494634, 17.021956681401335, -157.4690388764922)
    return ensemble_accuracy_array

def func_diversity(data):
    n, d = data  # x: N, y: D
    avg_data = np.sum(d)/n
    diversity = func_div((n, avg_data), 1.7608163972129824e-09, 1.5104552112861804, -6.525417189616139e-05, -2522.661603100801,
                    0.097812298684595, 0.593434375934323, -1.1861713921636314, 2522.9784618367407, -1.4699885825297731e-06)
    return diversity

def plot_3d(x,y,z, legend,xlabel,ylabel,zlabel,rst):
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='r', label=legend)
    ax.set_xlabel(xlabel, fontsize=12, rotation=50)
    ax.set_ylabel(ylabel, fontsize=12, rotation=-10)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.view_init(15, -150)
    x_major_locator = MultipleLocator(4)
    bx = plt.gca()
    bx.xaxis.set_major_locator(x_major_locator)
    ax.legend(fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    ax.tick_params(axis='z', labelsize=12)
    x_major_locator = MultipleLocator(20)
    y_major_locator = MultipleLocator(30000)
    bx = plt.gca()
    bx.xaxis.set_major_locator(x_major_locator)
    bx.yaxis.set_major_locator(y_major_locator)
    plt.tight_layout()
    sfig = plt.gcf()
    sfig.savefig(rst, format="pdf", bbox_inches='tight')
    plt.show()

def plotyy_2d(x,y1,y2,xlabel,marker1,ylabel1,legend1,marker2,ylabel2,legend2,rst):
    fig = plt.figure(figsize=(4, 3))
    plt.plot(x, y1, color=palette[0], marker=marker1)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel1, fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    ax2 = plt.twinx()
    ax2.plot(x, y2, color=palette[1], marker=marker2)
    ax2.set_ylabel(ylabel2, fontsize=12)
    fig.legend([legend1, legend2], bbox_to_anchor=(0.6, 0.8), fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    sfig = plt.gcf()
    sfig.savefig(rst, format="pdf", bbox_inches='tight')
    plt.show()

def plot_2d(x,y,xlabel,ylabel,marker,rst,xlim=None,ylim=None, y_major=None,rolling=None):
    if rolling is not None:
        average_y = []
        for ind in range(len(y) - rolling + 1):
            average_y.append(np.mean(y[ind:ind + rolling]))
        for ind in range(rolling - 1):
            average_y.insert(0, np.nan)
        y = average_y
    fig = plt.figure(figsize=(4, 3))
    plt.plot(x, y, color=palette[0], marker=marker)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if y_major is not None:
        y_major_locator = MultipleLocator(y_major)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plt.tight_layout()
    sfig = plt.gcf()
    sfig.savefig(rst, format="pdf", bbox_inches='tight')
    plt.show()


def plot_2d_mul_lines(x,y,xlabel,ylabel,marker,label,rst):
    fig = plt.figure(figsize=(4, 3))
    for i in range(len(y)):
        plt.plot(x, y[i], color=palette[i], marker=marker[i], label=label[i])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    # x_major_locator = MultipleLocator(4)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    sfig = plt.gcf()
    sfig.savefig(rst, format="pdf", bbox_inches='tight')
    plt.show()


with open('./results/mnist-ict-rst1.csv', 'a') as f:
    f.write("learner" + ',' + "datasize" + ',' + 'joint' + ',' + 'reward' + ',' + 'server payoff' + '\n')
    f.close()

with open('./results/mnist-ict-rst2.csv', 'a') as f:
    f.write("gamma" + ',' + "iter" + ',' + "total payoff" + ',' + 'num of learners' + ',' + 'Diversity' + ',' + 'Avg accuracy' + ',' + 'EL accuracy' + '\n')
    f.close()

with open('./results/mnist-ict-gamma.csv', 'a') as f:
    f.write("gamma" + ',' + "no. convergence" + ',' + 'no. learners' + ',' + "no. data size" + ',' + "total reward" + ',' + "total payoff" + ','+ 'Diversity' + ','  + 'Avg accuracy' + ',' + 'EL accuracy' + '\n')
    f.close()


n_n=20
n_d=1000
nn = np.array([i*5+min_n for i in range(n_n)])
dd = np.array([i*60+min_d for i in range(n_d)])
x, y = np.meshgrid(nn, dd)
x = x.reshape(1,20*1000)[0]
y = y.reshape(1,20*1000)[0]
ensemble_accuracy_array = np.zeros([len(x)])
for i in range(len(x)):
        d_list = np.array([y[i] for j in range(x[i])])
        ensemble_accuracy_array[i] = func_el_accu((x[i], d_list))

# plot_3d(x,y,ensemble_accuracy_array,'Evl',r'$N$',r'$D$','Accuracy','./results/mnist-df-accu-fitted.pdf')


cost_rank = cost.argsort()  # increasing cost indices
# d_candidate = np.array([min_d+i for i in range(D_max-min_d)])


num_convergence_iter = []
num_joint = []
data_size_learner1 = []
data_size_learner2 = []
avg_data_size = []
EL_total_reward = []
EL_server_payoff = []
EL_diversity = []
EL_avg_accuracy = []
EL_accuracy = []

num_learner_time = []
data_size_1_time = []
avg_data_size_time = []
data_size_10_time = []
total_reward_time = []
server_payoff_time = []
diversity_time = []
avg_accu_time = []
EL_accu_time = []

reward = np.ones(num_of_base_learners)
data_size = np.ones(num_of_base_learners) * 500
learner_decision = np.ones(num_of_base_learners)

for round in range(len(gamma_list)):
    # reward = np.ones(num_of_base_learners)
    # data_size = np.ones(num_of_base_learners) * 500
    # learner_decision = np.ones(num_of_base_learners)

    gamma = gamma_list[round]
    num_iter = 0
    total_server_payoff = 0
    total_server_payoff_old = 0
    while num_iter < 100 and (num_iter == 0 or total_server_payoff != total_server_payoff_old):
        num_iter += 1
        for n in range(num_of_base_learners):
            learner_index = cost_rank[n]
            d_candidate = np.array([min_d + i for i in range(D_max[learner_index] - min_d)])
            d_sum_except_n = 0
            accu_sum_except_n = 0.0
            n_join = 0
            for m in range(num_of_base_learners):
                if m != learner_index and learner_decision[m] == 1:
                    n_join += 1
                    d_sum_except_n += data_size[m]
                    accu = func_accu(data_size[m], 5.7952987719351795e-06, 0.00013345338910324258, -0.010678528614487103, 2.945455527694917e-07,
                                 1.000181238936544, 0.9734099561091535)
                    accu_sum_except_n += accu
            n_join += 1

            accu_candidate = func_accu(d_candidate, 5.7952987719351795e-06, 0.00013345338910324258,
                                       -0.010678528614487103, 2.945455527694917e-07,
                                       1.000181238936544, 0.9734099561091535)

            if n_join == 1:
                server_payoff = gamma * accu_candidate - cost[learner_index] * d_candidate + 1
                # determine D
                max_payoff_index = server_payoff.argsort()[len(server_payoff) - 1]
                max_payoff = server_payoff[max_payoff_index]
                best_d = d_candidate[max_payoff_index]
                data_size[learner_index] = best_d
                # determine R
                reward[learner_index] = cost[learner_index]*best_d
                learner_decision[learner_index] = 1

            else:
                d_avg_candidate = (d_candidate + d_sum_except_n) / n_join
                diversity_candidate = func_div((n_join,d_avg_candidate),1.7608163972129824e-09, 1.5104552112861804, -6.525417189616139e-05, -2522.661603100801,
                    0.097812298684595, 0.593434375934323, -1.1861713921636314, 2522.9784618367407, -1.4699885825297731e-06)
                avg_accu_candidate = (accu_sum_except_n+accu_candidate)/n_join
                el_accu_candidate = diversity_candidate+(avg_accu_candidate-1)/(n_join-1) + 1  #

                server_payoff = gamma*el_accu_candidate-cost[learner_index]*d_candidate

                # determine D
                max_payoff_index = server_payoff.argsort()[len(server_payoff)-1]
                max_payoff = server_payoff[max_payoff_index]
                best_d = d_candidate[max_payoff_index]
                data_size[learner_index] = best_d

                # determine R
                if n_join == 2:
                    el_accu_wo_n = accu_sum_except_n / (n_join-1)
                else:
                    diversity_wo_n = func_div((n_join-1,d_sum_except_n / (n_join-1)),1.7608163972129824e-09, 1.5104552112861804, -6.525417189616139e-05, -2522.661603100801,
                        0.097812298684595, 0.593434375934323, -1.1861713921636314, 2522.9784618367407, -1.4699885825297731e-06)
                    avg_accu_wo_n = accu_sum_except_n / (n_join-1)
                    el_accu_wo_n = diversity_wo_n+(avg_accu_wo_n-1)/(n_join-2) + 1  #
                server_payoff_wo_n = gamma*el_accu_wo_n

                server_payoff_dif = max_payoff - server_payoff_wo_n
                if n_join <= 2 or (server_payoff_dif >= 0 and max_payoff >= 0):
                    reward[learner_index] = cost[learner_index]* best_d
                    learner_decision[learner_index] = 1
                else:
                    reward[learner_index] = 0
                    learner_decision[learner_index] = 0
                    data_size[learner_index] = 0
                    n_join -= 1

            # print("-----learner" + str(learner_index) + ": datasize:" + str(best_d) + " join:" + str(learner_decision[learner_index])
            #       + " reward:" + str(reward[learner_index]) + " server payoff: " + str(max_payoff))

            with open('./results/mnist-ict-rst1.csv', 'a') as f:
                f.write(str(learner_index) + ',' + str(best_d) + ',' + str(learner_decision[learner_index]) + ',' +
                        str(reward[learner_index]) + ',' + str(max_payoff) + '\n')
                f.close()


        total_server_payoff_old = total_server_payoff
        total_server_payoff = 0

        n_join = np.sum(learner_decision)
        d_list = []
        for m in range(num_of_base_learners):
            if learner_decision[m] == 1:
                d_list.append(data_size[m])
        d_list = np.array(d_list)
        diversity = func_diversity((n_join, d_list))
        avg_accu = func_avg_accu((n_join, d_list))
        el_accu = func_el_accu((n_join, d_list))
        el_accu_real = func_el_accu_real_((n_join, d_list))
        total_server_payoff += gamma * el_accu
        for m in range(num_of_base_learners):
            if learner_decision[m] == 1:
                total_server_payoff -= reward[m]

        if gamma == gamma_list[0]:
            num_learner_time.append(n_join)
            data_size_1_time.append(data_size[0])
            avg_data_size_time.append(np.sum(data_size) / np.sum(learner_decision))
            data_size_10_time.append(data_size[4])
            total_reward_time.append([np.sum(reward)])
            server_payoff_time.append(total_server_payoff)
            diversity_time.append(diversity)
            avg_accu_time.append(avg_accu)
            EL_accu_time.append(el_accu)  # el_accu_real

        print("iter" + str(num_iter) + " total server payoff: " + str(total_server_payoff) + " # learners:" + str(n_join) + " diversity:" + str(diversity) + " accuracy:" + str(el_accu_real))
        with open('./results/mnist-ict-rst2.csv', 'a') as f:
            f.write(
                str(gamma) + ',' + str(num_iter) + ',' + str(total_server_payoff) + ',' + str(n_join) + ',' + str(diversity)
                + ',' + str(avg_accu) + ',' + str(el_accu) + '\n')
            f.close()


    if round == len(gamma_list)-1:
        x = []
        for n in range(num_of_base_learners):
            if learner_decision[n] == 1:
                x.append(n)

        cost_ = cost[x]
        data_size_= data_size[x]
        reward_ = reward[x]
        y = np.array([n for n in range(int(np.sum(learner_decision)))])

        plotyy_2d(y, cost_, data_size_, 'Base learner', "*", r'$\alpha_i+\beta_i$', r'$\alpha_i+\beta_i$',
                  "s", r'$D_i$', r'$D_i$', './results/mnist-ict-sgl-d.pdf')

        plotyy_2d(y, cost_, reward_, 'Base learner', "*", r'$\alpha_i+\beta_i$', r'$\alpha_i+\beta_i$',
                  "s", r'$R_i$', r'$R_i$', './results/mnist-ict-sgl-r.pdf')


    num_convergence_iter.append(num_iter)
    num_joint.append(n_join)
    data_size_learner1.append(data_size[0])
    data_size_learner2.append(data_size[4])
    avg_data_size.append(np.mean(data_size))
    EL_total_reward.append(np.sum(reward))
    EL_server_payoff.append(total_server_payoff)
    EL_diversity.append(diversity)
    EL_avg_accuracy.append(avg_accu)
    EL_accuracy.append(el_accu)  # el_accu_real

    with open('./results/mnist-ict-gamma.csv', 'a') as f:
        f.write(
            str(gamma) + ',' + str(num_iter) + ',' + str(n_join) + ',' + str(np.mean(data_size)) + ',' + str(np.sum(reward))
            + ',' + str(total_server_payoff) + ',' + str(diversity) + ',' + str(avg_accu) + ',' + str(el_accu) + '\n')
        f.close()

y=np.array(num_learner_time)
x = np.array([i+1 for i in range(len(num_learner_time))])
plot_2d(x,y,'Iteration','# participating learners',"*",'./results/mnist-ict-el-cov-num.pdf')

y1=np.array(data_size_1_time)
y2=np.array(data_size_10_time)
y3=np.array(avg_data_size_time)
x = np.array([i+1 for i in range(len(avg_data_size_time))])
plot_2d_mul_lines(x,[y1,y2,y3],'Iteration','Data size',["*","s","o"],
                  ['Learner 1','Learner 10','Average'],'./results/mnist-ict-el-cov-datasize.pdf')

y=np.array(total_reward_time)
x = np.array([i+1 for i in range(len(total_reward_time))])
plot_2d(x,y,'Iteration','Total reward',"*",'./results/mnist-ict-el-cov-reward.pdf')


y=np.array(server_payoff_time)
x = np.array([i+1 for i in range(len(server_payoff_time))])
plot_2d(x,y,'Iteration','Server payoff',"*",'./results/mnist-ict-el-cov-payoff.pdf')

y=np.array(diversity_time)
x = np.array([i+1 for i in range(len(diversity_time))])
plot_2d(x,y,'Iteration','Diversity',"*",'./results/mnist-ict-el-cov-diversity.pdf')


y=np.array(avg_accu_time)
x = np.array([i+1 for i in range(len(avg_accu_time))])
plot_2d(x,y,'Iteration','Average accuracy',"*",'./results/mnist-ict-el-cov-avgaccu.pdf')


y=np.array(EL_accu_time)
x = np.array([i+1 for i in range(len(EL_accu_time))])
plot_2d(x,y,'Iteration','Ensemble accuracy',"*",'./results/mnist-ict-el-cov-elaccu.pdf')


x = np.array(gamma_list)
y = np.array(num_convergence_iter)
plot_2d(x,y,r'$\gamma$','# iterations',".",'./results/mnist-ict-el-cov.pdf', y_major=1)

x = np.array(gamma_list)
y = np.array(num_joint)
plot_2d(x,y,r'$\gamma$','# participating learners',".",'./results/mnist-ict-el-num-learner.pdf',y_major=4)


x = np.array(gamma_list)
y1 = np.array(data_size_learner1)
y2 = np.array(data_size_learner2)
y3 = np.array(avg_data_size)
plot_2d_mul_lines(x,[y1,y2,y3],r'$\gamma$','Data size',[".","s","o"],
                  ['Learner 1','Learner 10','Average'],'./results/mnist-ict-el-d.pdf')

x = np.array(gamma_list)
y = np.array(EL_total_reward)
plot_2d(x,y,r'$\gamma$','Total reward',".",'./results/mnist-ict-el-reward.pdf')#,xlim=(1,6000))


x = np.array(gamma_list)
y = np.array(EL_server_payoff)
plot_2d(x,y,r'$\gamma$','Server payoff',".",'./results/mnist-ict-el-payoff.pdf')#,xlim=(1,6000))

x = np.array(gamma_list)
y = np.array(EL_diversity)+np.array([0.000006 for i in range(len(EL_diversity))])
plot_2d(x,y,r'$\gamma$','Diversity',".",'./results/mnist-ict-el-div.pdf', rolling=5)#,xlim=(1,6000))

x = np.array(gamma_list)
y = np.array(EL_avg_accuracy)
plot_2d(x,y,r'$\gamma$','Average accuracy',".",'./results/mnist-ict-el-avgaccu.pdf')#,xlim=(1,6000))

x = np.array(gamma_list)
y = np.array(EL_accuracy)
plot_2d(x,y,r'$\gamma$','Ensemble accuracy',".",'./results/mnist-ict-el-elaccu.pdf')#,xlim=(1,6000))
