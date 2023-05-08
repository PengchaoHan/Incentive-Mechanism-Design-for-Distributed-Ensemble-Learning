import torch


def majority_voting(num_based_learners, model_list, data_test_loader, num_classes, device):
    total_correct = 0
    accuracy_list=[0]*num_based_learners
    with torch.no_grad():
        for _, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)
            voting_result = torch.zeros(images.shape[0], num_classes).to(device)
            for n in range(num_based_learners):
                model_list[n].eval()
                output = model_list[n](images)
                pred = output.data.max(1)[1]
                for i in range(images.shape[0]):
                    voting_result[i][pred[i]] += 1
                accuracy_list[n]+=pred.eq(labels.data.view_as(pred)).sum()
            predv = voting_result.argmax(dim=1)
            total_correct += predv.eq(labels.data.view_as(predv)).sum()

    acc = float(total_correct) / len(data_test_loader.dataset)
    for i in range(num_based_learners):
        accuracy_list[i]=float(accuracy_list[i]) / len(data_test_loader.dataset)

    return acc, accuracy_list


def q_statistics(num_base_learners, model_list, data_test_loader, device):
    pred_results = []
    with torch.no_grad():
        for n in range(num_base_learners):
            correct_list=[]
            model_list[n].eval()
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = model_list[n](images)
                pred = output.data.max(1)[1]
                correct_list.append(pred.eq(labels.data.view_as(pred)))
            pred_results.append(torch.cat(correct_list))

    q = torch.zeros([num_base_learners, num_base_learners]).to(device)
    for i in range(num_base_learners):
        for j in range(num_base_learners):
            if i <= j:
                n11 = (pred_results[i]*pred_results[j]).sum()
                n00 = ((~pred_results[i])*(~pred_results[j])).sum()
                n10 = (pred_results[i]*(~pred_results[j])).sum()
                n01 = ((~pred_results[i])*pred_results[j]).sum()
                q[i][j] = (float(n11*n00)-float(n10*n01)) / (float(n11*n00)+float(n10*n01))
                q[j][i] = q[i][j]
    q_avg = q.sum()/(num_base_learners*num_base_learners)
    return q, q_avg

def double_fault(num_base_learners, model_list, data_test_loader, device, data_sample_index):
    # record_used_sample = torch.zeros([num_base_learners,len(data_test_loader.dataset)])
    # for n in range(num_base_learners):
    #     record_used_sample[n][data_sample_index[n]] = 1
    batch_size = 32
    with torch.no_grad():
        sum_square = 0
        for batch_index, (images, labels) in enumerate(data_test_loader):
            images = images.numpy()
            labels = labels.to(device)
            for smp_index in range(len(images)):
                num_wrong_prediction = 0
                for n in range(num_base_learners):
                    if (batch_index*batch_size+smp_index) in data_sample_index[n]:  #record_used_sample[n][batch_index*len(images)+smp_index] == 1:
                        image = images[smp_index]
                        image = torch.from_numpy(image)
                        image = torch.unsqueeze(image, 0)
                        image = image.to(device)
                        model_list[n].to(device)
                        model_list[n].eval()
                        output = model_list[n](image)
                        pred = output.data.max(1)[1]
                        if not pred.eq(labels[smp_index].data.view_as(pred)):
                            num_wrong_prediction += 1
                sum_square += (num_wrong_prediction*num_wrong_prediction)
    df_div = float(1 / len(data_test_loader.dataset) * sum_square)
    df_diva = float(1 / (num_base_learners * (num_base_learners - 1)) * df_div)
    return df_div, df_diva

def q_statistics_v2(num_based_learners,model_list,data_test_loader, device):
    predict_wrong=0
    result=0
    data_test_size=len(data_test_loader.dataset)
    pred_results = []

    with torch.no_grad():
        for n in range(num_based_learners):
            correct_list=[]
            model_list[n].eval()
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = model_list[n](images)
                pred = output.data.max(1)[1]
                correct_list.append(pred.eq(labels.data.view_as(pred)))
            pred_results.append(torch.cat(correct_list))
        
    for i in range(len(pred_results)):
        Id = 0
        for j in range(num_based_learners):
            if (~pred_results[j][i]):
                Id += 1
        predict_wrong += Id **2
        Id = 0
    result = float(1/(num_based_learners*(num_based_learners-1)*data_test_size)*predict_wrong)
    return result


def mse_calculation(diversity_matrix,accuracy_list,num_of_base_learners):
    result_d=1
    result_a=1
    count=1
    result=[1]*num_of_base_learners
    for k in range(1,num_of_base_learners+1):
        for i in range (k):
            for j in range (k):
                if i <= j : break
                else:
                    result_d*=(1-diversity_matrix[i][j]) #一旦一个模型一样就作废
        for num in accuracy_list:
            if k>1:
                result_a*=(1-num)
                count+=1
                if count==k+1: 
                    count=1
                    break
            else:
                break
        result[k-1]=result_d**(1/((num_of_base_learners-1)*num_of_base_learners))*result_a**(1/num_of_base_learners)
        result_a=1
        result_d=1
    return result
