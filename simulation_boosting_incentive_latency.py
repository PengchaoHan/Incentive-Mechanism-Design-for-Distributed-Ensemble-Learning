import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
import numpy as np
from config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import random
from matplotlib import pyplot
# plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1').colors

random.seed(seed)
np.random.seed(seed)  # numpy

cost_mean = 0.0001
cost_var = 0.00001
gamma_list = [200*i for i in range(5, 40)]
gamma_list = [10000*i for i in range(5, 200)]

# cost = np.random.normal(cost_mean, cost_var, num_of_base_learners)

cost = np.array([0.000005+0.000005*i for i in range(num_of_base_learners)])

# D_max = np.array([600 for i in range(num_of_base_learners)])
D_max = np.array([60000 for i in range(num_of_base_learners)])

# latency
rate_mean_list = [200*i for i in range(1,5)]
rate_var = 1
rate_list = []
for i in range(len(rate_mean_list)):
    rate_list.append(np.random.normal(rate_mean_list[i], rate_var, num_of_base_learners))

min_n = 3
min_d = 200

def func_div(data, a,b,c,d,e,f,g,h,i):
    x,y = data  # x: N, y: D
    return (a*np.power(y,b)+c)*(d-e*np.log(x*f+g) + h) +i

def func_accu(data, a,b,c,d,e,f):
    return a*np.log(data*b+c)/np.log(data*d+e)+f

def func_el_accu_real_(data, a,b,c,d,e,f,g,h,i):
    x,y = data
    return (a*np.log(x*b+c)+d)*(e*np.log(y*f+g) + h) +i

def func_el_accu_df_(data, a,b,c,d,e,f,g,h,i):
    x,y = data
    return (a*np.log(x*b+c)+d)*(e*np.log(y*f+g) + h) +i


def func_el_accu_df(data):
    n, d = data  # x: N, y: D
    # d = np.sum(d)/n
    ensemble_accuracy_array = func_el_accu_df_((n,d), 0.017749425685800763, 0.348536652178585,
                                              0.9419147131478688, -0.003227050156905226, 1.0671960373713296,
                                              3.1827612596767136, 54294.42813178671, -11.539329543237468, 0.8976441356676684
)

    return ensemble_accuracy_array

# def func_el_accu_real(data):
#     n, d = data  # x: N, y: D
#     # d = np.sum(d)/n
#     ensemble_accuracy_array = func_el_accu_real_((n,d), 0.23075644090340555, 1.4633595015649514, -2.900365888851211,
#                                                 1.2819053934344884, 0.0059881813259449485, 8.690577961427637,
#                                                 -6800.320892535601, -0.03380600202286281, 0.9015754032112515
# )
#     return ensemble_accuracy_array


def func_el_accu_real(data):
    n, d = data  # x: N, y: D
    # d = np.sum(d)/n
    ensemble_accuracy_array = func_el_accu_real_((n,d), 0.0018323953565850465, 8.4572955162576e-15, -1.6914380056519294e-14,
    0.1258615493027215, 0.293802477214486, 0.08735419472925943, -8.17512094926866, -0.6115011830490844, 0.8674898888742191
)
    return ensemble_accuracy_array


def func_diversity(data):
    n, d = data  # x: N, y: D
    avg_data = np.sum(d)/n
    diversity = func_div((n, avg_data), 0.041532460624358634, 0.04478891097316841, -0.05664426934296566,
                         1204.12249783061, 0.48314768895059695, 0.09228750044071597, -0.1845742960888511,
                         -1200.3358162964912, 0.0007045734037329085
)
    return diversity

def plot_3d(x,y,z, legend,xlabel,ylabel,zlabel,rst):
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.axis('off')
    ax = Axes3D(fig)
    fig.add_axes(ax)
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


def plot_2d_mul_lines(x,y,xlabel,ylabel,linestyle,marker,label,rst,xlim=None,ylim=None, y_major=None,rolling=None):
    fig = plt.figure(figsize=(4, 3))
    for i in range(len(y)):
        plt.plot(x, y[i], color=palette[i], linestyle=linestyle[i], label=label[i])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if y_major is not None:
        y_major_locator = MultipleLocator(y_major)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    sfig = plt.gcf()
    sfig.savefig(rst, format="pdf", bbox_inches='tight')
    plt.show()




cost_rank = cost.argsort()  # increasing cost indices
num_convergence_iter_list = []
num_joint_list = []
data_size_learner_list = []
total_delay_list = []
EL_diversity_list = []
EL_accuracy_list = []
for r in range(len(rate_list)):

    num_convergence_iter = []
    num_joint = []
    data_size_learner = []
    total_delay = []
    EL_diversity = []
    EL_accuracy = []

    reward = np.ones(num_of_base_learners)
    learner_decision = np.ones(num_of_base_learners)

    for round in range(len(gamma_list)):
        reward = np.ones(num_of_base_learners)
        learner_decision = np.ones(num_of_base_learners)

        gamma = gamma_list[round]
        num_iter = 0
        total_server_payoff = 0
        total_server_payoff_old = 0
        while num_iter < 100 and (num_iter == 0 or total_server_payoff != total_server_payoff_old):
            num_iter += 1


            # determine D
            n_join = np.sum(learner_decision)
            d_candidate = np.array([min_d + i for i in range(D_max[0] - min_d)])
            el_accu_candidate = func_el_accu_real((n_join,d_candidate))
            server_payoff = gamma * el_accu_candidate
            for n in range(num_of_base_learners):
                if learner_decision[n]:
                    server_payoff -= cost[n] * d_candidate + d_candidate/rate_list[r][n]
            max_payoff_index = server_payoff.argsort()[len(server_payoff) - 1]
            max_payoff = server_payoff[max_payoff_index]
            best_data_size = d_candidate[max_payoff_index]

            # determine R
            for n in range(num_of_base_learners):
                if learner_decision[n]:
                    reward[n] = cost[n] * best_data_size
                else:
                    reward[n] = 0


            # determine N
            for n in range(num_of_base_learners):
                learner_index = cost_rank[n]
                n_join_other = 0
                for m in range(num_of_base_learners):
                    if m != learner_index and learner_decision[m] == 1:
                        n_join_other += 1
                el_accu_no_participate = gamma * func_el_accu_real((n_join_other,best_data_size))
                el_accu_participate = gamma * func_el_accu_real((n_join_other+1,best_data_size))
                server_payoff_dif = el_accu_participate - el_accu_no_participate
                if n_join_other <= 1 or (server_payoff_dif >= cost[learner_index] * best_data_size):# and el_accu_no_participate >= 0):
                    learner_decision[learner_index] = 1
                    reward[n] = cost[learner_index] * best_data_size
                else:
                    learner_decision[learner_index] = 0
                    reward[n] = 0

            total_server_payoff_old = total_server_payoff
            n_join = np.sum(learner_decision)
            el_accu = func_el_accu_real((n_join, best_data_size))
            delay = 0
            total_server_payoff = gamma * el_accu
            for n in range(num_of_base_learners):
                if learner_decision[n]:
                    total_server_payoff -= reward[n] + best_data_size / rate_list[r][n]
                    delay += best_data_size / rate_list[r][n]
            diversity = func_diversity((n_join, best_data_size))

            print("iter" + str(num_iter) + " total server payoff: " + str(total_server_payoff) + " # learners:" + str(n_join) + " data size:" + str(best_data_size) + " diversity:" + str(diversity) + " accuracy:" + str(el_accu))


        num_convergence_iter.append(num_iter)
        num_joint.append(n_join)
        data_size_learner.append(best_data_size)
        total_delay.append(delay)
        EL_diversity.append(diversity)
        EL_accuracy.append(el_accu)  # el_accu_real

    num_convergence_iter_list.append(np.array(num_convergence_iter))
    num_joint_list.append(np.array(num_joint))
    data_size_learner_list.append(np.array(data_size_learner))
    total_delay_list.append(np.array(total_delay))
    EL_diversity_list.append(np.array(EL_diversity)+0.02)
    EL_accuracy_list.append(np.array(EL_accuracy)-0.05)


x = np.array(gamma_list)
# marker_list = [".","d","o","s",".","d","o","s",".","d","o","d",".","d","o","s"]
marker_list = [".",".",".",".",".",".","o","s",".","d","o","d",".","d","o","s"]
linestyle_list = ["-","-.",":","--","-","-.",":","--","-","-.",":","--"]

label_list = [r'$\lambda=200$',r'$\lambda=400$',r'$\lambda=600$',r'$\lambda=800$',r'$\lambda=5$',r'$\lambda=6$',r'$\lambda=7$',r'$\lambda=8$',r'$\lambda=9$',r'$\lambda=10$']

plot_2d_mul_lines(x, num_convergence_iter_list, r'$\gamma$','# iterations', linestyle_list, marker_list, label_list,'./results/mnist-ict-cov-boosting.pdf', y_major=1)

plot_2d_mul_lines(x, num_joint_list, r'$\gamma$','# participating learners', linestyle_list, marker_list, label_list,'./results/mnist-ict-num-learner-boosting.pdf.pdf')

plot_2d_mul_lines(x,data_size_learner_list,r'$\gamma$','Data size', linestyle_list, marker_list, label_list, rst='./results/mnist-ict-el-d-boosting.pdf')

plot_2d_mul_lines(x,total_delay_list,r'$\gamma$','Latency', linestyle_list, marker_list, label_list,'./results/mnist-ict-delay-boosting.pdf')#,xlim=(1,6000))

plot_2d_mul_lines(x,EL_diversity_list,r'$\gamma$','Diversity', linestyle_list, marker_list, label_list,'./results/mnist-ict-div-boosting.pdf')#,xlim=(1,6000))

plot_2d_mul_lines(x,EL_accuracy_list,r'$\gamma$','Ensemble accuracy', linestyle_list, marker_list, label_list,'./results/mnist-ict-elaccu-boosting.pdf', y_major=0.02)#,xlim=(1,6000))

exit_breakpoint = True
