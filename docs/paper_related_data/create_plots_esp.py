import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
import scipy.stats
import statistics


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


    arr_list  = []
    yerr = np.zeros((2, len(list_values)))
    for i in range(0, len(list_values)):
        lower_val, upper_val = mean_confidence_interval(list_values[i])
        yerr[0][i] = lower_val
        yerr[1][i] = upper_val
        
    return yerr

def calulte_standard_dev(list_values):
    std_values = []
    for i in range(len(list_values)):
        std_values.append(statistics.stdev(list_values[i]))
    return std_values
  
def calculate_average(values):
    return np.mean(values)

# mnist_cnn_10_clients= [117, 134, 159, 110, 159]
# mnist_cnn_30_clients= [73, 73, 47, 69, 77]
# mnist_cnn_50_clients= [69, 69, 56, 47, 63]
# mnist_cnn_70_clients= [47, 47, 45, 57, 69]
# mnist_cnn_100_clients= [56, 64, 59, 54, 49]

mnist_cnn_10_clients_adam_5 = [135, 116]
mnist_cnn_10_clients_adam_10 = [135, 116]
mnist_cnn_10_clients_ndam_5 = [125, 145]
mnist_cnn_10_clients_ndam_10 = [123, 109]
mnist_cnn_10_clients_sgd_5 = [116, 140]
mnist_cnn_10_clients_sgd_10 = [162, 150]

mnist_cnn_30_clients_adam_5 = [56, 64, 69]
mnist_cnn_30_clients_adam_10 = [68, 51]
mnist_cnn_30_clients_ndam_5 = [85, 78]
mnist_cnn_30_clients_ndam_10 = [71, 65]
mnist_cnn_30_clients_sgd_5 =  [108, 82]
mnist_cnn_30_clients_sgd_10 =  [102, 105]

mnist_cnn_50_clients_adam_5 = [76, 56]
mnist_cnn_50_clients_adam_10 = [79, 47]
mnist_cnn_50_clients_ndam_5 = [75, 52]
mnist_cnn_50_clients_ndam_10 = [60, 55]
mnist_cnn_50_clients_sgd_5 = [86, 87]
mnist_cnn_50_clients_sgd_10 = [81, 99]

mnist_cnn_70_clients_adam_5 = [69, 65]
mnist_cnn_70_clients_adam_10 = [44, 54]
mnist_cnn_70_clients_ndam_5 = [67, 49]
mnist_cnn_70_clients_ndam_10 = [46, 48]
mnist_cnn_70_clients_sgd_5 = [128, 99]
mnist_cnn_70_clients_sgd_10 = [97, 106]

mnist_cnn_100_clients_adam_5 = [60, 50]
mnist_cnn_100_clients_adam_10 =[47, 48]
mnist_cnn_100_clients_ndam_5 = [45, 54]
mnist_cnn_100_clients_ndam_10 =[45, 50]
mnist_cnn_100_clients_sgd_5 = [125, 100]
mnist_cnn_100_clients_sgd_10 = [84, 131]



mnist_cnn_values_adam_5 = []
mnist_cnn_values_adam_10 = []
mnist_cnn_values_ndam_5 = []
mnist_cnn_values_ndam_10 = []
mnist_cnn_values_sgd_5 = []
mnist_cnn_values_sgd_10 = []
# mnist_cnn_values = []


mnist_cnn_values_adam_5.append(mnist_cnn_10_clients_adam_5)
mnist_cnn_values_adam_10.append(mnist_cnn_10_clients_adam_10)
mnist_cnn_values_ndam_5.append(mnist_cnn_10_clients_ndam_5)
mnist_cnn_values_ndam_10.append(mnist_cnn_10_clients_ndam_10)
mnist_cnn_values_sgd_5.append(mnist_cnn_10_clients_sgd_5)
mnist_cnn_values_sgd_10.append(mnist_cnn_10_clients_sgd_10)

mnist_cnn_values_adam_5.append(mnist_cnn_30_clients_adam_5)
mnist_cnn_values_adam_10.append(mnist_cnn_30_clients_adam_10)
mnist_cnn_values_ndam_5.append(mnist_cnn_30_clients_ndam_5)
mnist_cnn_values_ndam_10.append(mnist_cnn_30_clients_ndam_10)
mnist_cnn_values_sgd_5.append(mnist_cnn_30_clients_sgd_5)
mnist_cnn_values_sgd_10.append(mnist_cnn_30_clients_sgd_10)

mnist_cnn_values_adam_5.append(mnist_cnn_50_clients_adam_5)
mnist_cnn_values_adam_10.append(mnist_cnn_50_clients_adam_10)
mnist_cnn_values_ndam_5.append(mnist_cnn_50_clients_ndam_5)
mnist_cnn_values_ndam_10.append(mnist_cnn_50_clients_ndam_10)
mnist_cnn_values_sgd_5.append(mnist_cnn_50_clients_sgd_5)
mnist_cnn_values_sgd_10.append(mnist_cnn_50_clients_sgd_10)

mnist_cnn_values_adam_5.append(mnist_cnn_70_clients_adam_5)
mnist_cnn_values_adam_10.append(mnist_cnn_70_clients_adam_10)
mnist_cnn_values_ndam_5.append(mnist_cnn_70_clients_ndam_5)
mnist_cnn_values_ndam_10.append(mnist_cnn_70_clients_ndam_10)
mnist_cnn_values_sgd_5.append(mnist_cnn_70_clients_sgd_5)
mnist_cnn_values_sgd_10.append(mnist_cnn_70_clients_sgd_10)

mnist_cnn_values_adam_5.append(mnist_cnn_100_clients_adam_5)
mnist_cnn_values_adam_10.append(mnist_cnn_100_clients_adam_10)
mnist_cnn_values_ndam_5.append(mnist_cnn_100_clients_ndam_5)
mnist_cnn_values_ndam_10.append(mnist_cnn_100_clients_ndam_10)
mnist_cnn_values_sgd_5.append(mnist_cnn_100_clients_sgd_5)
mnist_cnn_values_sgd_10.append(mnist_cnn_100_clients_sgd_10)
# mnist_cnn_values.append(mnist_cnn_10_clients)
# mnist_cnn_values.append(mnist_cnn_30_clients)
# mnist_cnn_values.append(mnist_cnn_50_clients)
# mnist_cnn_values.append(mnist_cnn_70_clients)
# mnist_cnn_values.append(mnist_cnn_100_clients)

# mnist_nn_10_clients= [272, 217, 332, 212, 354]
# mnist_nn_30_clients= [165, 167, 140, 176, 161]
# mnist_nn_50_clients= [190, 141, 123, 122, 170]
# mnist_nn_70_clients= [168, 126, 135, 221, 121]
# mnist_nn_100_clients= [210, 194, 211, 237, 209]

mnist_nn_10_clients_adam_5 = [212, 260] 
mnist_nn_10_clients_adam_10 = [192, 161] 

mnist_nn_10_clients_ndam_5 = [246, 248] 
mnist_nn_10_clients_ndam_10 = [177, 186] 
mnist_nn_10_clients_sgd_5 = [306, 211] 
mnist_nn_10_clients_sgd_10 = [273, 212]

mnist_nn_30_clients_adam_5 = [170, 83] 
mnist_nn_30_clients_adam_10 =  [106, 88]
mnist_nn_30_clients_ndam_5 =  [200, 146]
mnist_nn_30_clients_ndam_10 = [114, 89]
mnist_nn_30_clients_sgd_5 =  [260, 159]
mnist_nn_30_clients_sgd_10 = [179, 165]

mnist_nn_50_clients_adam_5 = [158, 79] 
mnist_nn_50_clients_adam_10 =  [107, 65]
mnist_nn_50_clients_ndam_5 =  [203, 100]
mnist_nn_50_clients_ndam_10 =  [81, 89]
mnist_nn_50_clients_sgd_5 =  [258, 226]
mnist_nn_50_clients_sgd_10 = [177, 228]

mnist_nn_70_clients_adam_5 = [161, 153]
mnist_nn_70_clients_adam_10 =  [64, 47]
mnist_nn_70_clients_ndam_5 =  [237, 143]
mnist_nn_70_clients_ndam_10 = [107, 83]
mnist_nn_70_clients_sgd_5= [269, 172]
mnist_nn_70_clients_sgd_10 = [210, 135]

mnist_nn_100_clients_adam_5 = [386, 103]
mnist_nn_100_clients_adam_10 = [61, 54]
mnist_nn_100_clients_ndam_5 = [277, 142]
mnist_nn_100_clients_ndam_10 = [62, 71]
mnist_nn_100_clients_sgd_5 = [252, 167]
mnist_nn_100_clients_sgd_10 = [169, 250]


mnist_nn_values_adam_5 = []
mnist_nn_values_adam_10 = []
mnist_nn_values_ndam_5 = []
mnist_nn_values_ndam_10 = []
mnist_nn_values_sgd_5 = []
mnist_nn_values_sgd_10 = []


# mnist_nn_values = []
# mnist_nn_values.append(mnist_nn_10_clients)
# mnist_nn_values.append(mnist_nn_30_clients)
# mnist_nn_values.append(mnist_nn_50_clients)
# mnist_nn_values.append(mnist_nn_70_clients)
# mnist_nn_values.append(mnist_nn_100_clients)
mnist_nn_values_adam_5.append(mnist_nn_10_clients_adam_5)
mnist_nn_values_adam_10.append(mnist_nn_10_clients_adam_10)
mnist_nn_values_ndam_5.append(mnist_nn_10_clients_ndam_5)
mnist_nn_values_ndam_10.append(mnist_nn_10_clients_ndam_10)
mnist_nn_values_sgd_5.append(mnist_nn_10_clients_sgd_5)
mnist_nn_values_sgd_10.append(mnist_nn_10_clients_sgd_10)


mnist_nn_values_adam_5.append(mnist_nn_30_clients_adam_5)
mnist_nn_values_adam_10.append(mnist_nn_30_clients_adam_10)
mnist_nn_values_ndam_5.append(mnist_nn_30_clients_ndam_5)
mnist_nn_values_ndam_10.append(mnist_nn_30_clients_ndam_10)
mnist_nn_values_sgd_5.append(mnist_nn_30_clients_sgd_5)
mnist_nn_values_sgd_10.append(mnist_nn_30_clients_sgd_10)


mnist_nn_values_adam_5.append(mnist_nn_50_clients_adam_5)
mnist_nn_values_adam_10.append(mnist_nn_50_clients_adam_10)
mnist_nn_values_ndam_5.append(mnist_nn_50_clients_ndam_5)
mnist_nn_values_ndam_10.append(mnist_nn_50_clients_ndam_10)
mnist_nn_values_sgd_5.append(mnist_nn_50_clients_sgd_5)
mnist_nn_values_sgd_10.append(mnist_nn_50_clients_sgd_10)


mnist_nn_values_adam_5.append(mnist_nn_70_clients_adam_5)
mnist_nn_values_adam_10.append(mnist_nn_70_clients_adam_10)
mnist_nn_values_ndam_5.append(mnist_nn_70_clients_ndam_5)
mnist_nn_values_ndam_10.append(mnist_nn_70_clients_ndam_10)
mnist_nn_values_sgd_5.append(mnist_nn_70_clients_sgd_5)
mnist_nn_values_sgd_10.append(mnist_nn_70_clients_sgd_10)

mnist_nn_values_adam_5.append(mnist_nn_100_clients_adam_5)
mnist_nn_values_adam_10.append(mnist_nn_100_clients_adam_10)
mnist_nn_values_ndam_5.append(mnist_nn_100_clients_ndam_5)
mnist_nn_values_ndam_10.append(mnist_nn_100_clients_ndam_10)
mnist_nn_values_sgd_5.append(mnist_nn_100_clients_sgd_5)
mnist_nn_values_sgd_10.append(mnist_nn_100_clients_sgd_10)










font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 16}
matplotlib.rcParams.update({'font.size': 14})
# matplotlib.rcParams.update({'font.weight': 'bold'})
# matplotlib.rc('font', **font)

# NUM_TESTS = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:brown']
NUM_TESTS = ['tab:orange', 'tab:blue', 'tab:green']
Grid_Sizes = ['10', '30', '50', '70', '100']

# NAME_TESTS = ['MNIST-CNN, local-epochs=5', 'MNIST-NN']
# NAME_TESTS = ['MNIST-CNN, local-epochs=5, opt=adam', 'MNIST-CNN, local-epochs=10, opt=adam', 'MNIST-CNN, local-epochs=5, opt=ndam', 'MNIST-CNN, local-epochs=10, opt=ndam', 'MNIST-CNN, local-epochs=5, opt=sgd', 'MNIST-CNN, local-epochs=10, opt=sgd']
# NAME_TESTS_5_cnn = ['MNIST-CNN, lr=1e-3, opt=adam',  'MNIST-CNN, lr=1e-3, opt=ndam', 'MNIST-CNN, lr=0.1, opt=sgd']
NAME_TESTS_5_cnn = ['lr=1e-3, opt=adam',  'lr=1e-3, opt=ndam', 'lr=0.1, opt=vanilla-sgd']
NAME_TESTS_10_cnn = ['lr=1e-3, opt=adam', 'lr=1e-3, opt=ndam',  'lr=0.1, opt=vanilla-sgd']
# NAME_TESTS = ['MNIST-NN, local-epochs=5, opt=adam', 'MNIST-NN, local-epochs=10, opt=adam', 'MNIST-NN, local-epochs=5, opt=ndam', 'MNIST-NN, local-epochs=10, opt=ndam', 'MNIST-NN, local-epochs=5, opt=sgd', 'MNIST-NN, local-epochs=10, opt=sgd']
# NAME_TESTS = ['MNIST-NN, local-epochs=5, opt=adam', 'MNIST-NN, local-epochs=10, opt=adam', 'MNIST-NN, local-epochs=5, opt=ndam', 'MNIST-NN, local-epochs=10, opt=ndam', 'MNIST-NN, local-epochs=5, opt=sgd', 'MNIST-NN, local-epochs=10, opt=sgd']
# NAME_TESTS_5_nn = ['MNIST-NN, local-epochs=5, opt=adam',  'MNIST-NN, local-epochs=5, opt=ndam', 'MNIST-NN, local-epochs=5, opt=sgd']
NAME_TESTS_5_nn = ['lr=1e-3, opt=adam',  'lr=1e-3, opt=ndam', 'lr=0.1, opt=vanilla-sgd']
# NAME_TESTS_10_nn = ['MNIST-NN, local-epochs=10, opt=adam',  'MNIST-NN, local-epochs=10, opt=ndam', 'MNIST-NN, local-epochs=10, opt=sgd']
NAME_TESTS_10_nn = ['lr=1e-3, opt=adam',  'lr=1e-3, opt=ndam', 'lr=0.1, opt=vanilla-sgd']


BAR_WIDTH = 1/(len(NUM_TESTS)*2)
bar_positions = []

for i in range(0, len(NUM_TESTS)):
    if i == 0:
        bar_positions.append(np.arange(len(Grid_Sizes)))
    else:
        bar_positions.append(np.array([x + BAR_WIDTH for x in bar_positions[i-1]]))

bar_positions = np.array(bar_positions)
bar_positions = np.transpose(bar_positions)


bar_values_cnn_5= []
bar_values_cnn_10= []
bar_values_nn_5= []
bar_values_nn_10= []

for i in range(0, len(Grid_Sizes)):
    bar_values_cnn_5.append(calculate_average(mnist_cnn_values_adam_5[i]))
    bar_values_nn_5.append(calculate_average(mnist_nn_values_adam_5[i]))

    bar_values_cnn_10.append(calculate_average(mnist_cnn_values_adam_10[i]))
    bar_values_nn_10.append(calculate_average(mnist_nn_values_adam_10[i]))

    # bar_values.append(calculate_average(mnist_cnn_values_adam_10[i]))
    bar_values_cnn_5.append(calculate_average(mnist_cnn_values_ndam_5[i]))
    bar_values_nn_5.append(calculate_average(mnist_nn_values_ndam_5[i]))

    bar_values_cnn_10.append(calculate_average(mnist_cnn_values_ndam_10[i]))
    bar_values_nn_10.append(calculate_average(mnist_nn_values_ndam_10[i]))

    
    # bar_values.append(calculate_average(mnist_cnn_values_ndam_10[i]))
    bar_values_cnn_5.append(calculate_average(mnist_cnn_values_sgd_5[i]))
    bar_values_nn_5.append(calculate_average(mnist_nn_values_sgd_5[i]))
    # bar_values.append(calculate_average(mnist_cnn_values_sgd_10[i]))
    bar_values_cnn_10.append(calculate_average(mnist_cnn_values_sgd_10[i]))
    bar_values_nn_10.append(calculate_average(mnist_nn_values_sgd_10[i]))

    # bar_values.append(calculate_average(mnist_nn_values_adam_5[i]))
    # bar_values.append(calculate_average(mnist_nn_values_adam_10[i]))
    # bar_values.append(calculate_average(mnist_nn_values_ndam_5[i]))
    # bar_values.append(calculate_average(mnist_nn_values_ndam_10[i]))
    # bar_values.append(calculate_average(mnist_nn_values_sgd_5[i]))
    # bar_values.append(calculate_average(mnist_nn_values_sgd_10[i]))
    # bar_values.append(calculate_average(mnist_nn_values[i]))


# values =[]
# values.append(mnist_cnn_values)
# values.append(mnist_nn_values)

plt.figure(figsize=(6, 4))
plt.rc('axes', axisbelow=True)
for (i, color) in enumerate(NUM_TESTS):
    # print(np.array(bar_values).reshape(5,2).transpose()[i])
    # print(bar_positions.transpose()[i])
    # plt.bar(bar_positions.transpose()[i], np.array(bar_values).reshape(5,2).transpose()[i], color=color, width=BAR_WIDTH, edgecolor='white', label=NAME_TESTS[i], yerr=calulte_standard_dev(values[i]), capsize=3)
    plt.bar(bar_positions.transpose()[i], np.array(bar_values_cnn_5).reshape(5,len(NUM_TESTS)).transpose()[i], color=color, width=BAR_WIDTH, edgecolor='white', label=NAME_TESTS_5_cnn[i])



plt.xlabel('Number of Clients')
plt.ylabel('Communication Rounds')
plt.xticks([bar_positions + BAR_WIDTH for bar_positions in range(len(Grid_Sizes))], ['10', '30', '50', '70', '100'])
#plt.legend(fontsize='small')
plt.ylim([0, 160])
plt.grid(linestyle='-.', linewidth=1,  color="0.9")
plt.savefig("comm_rounds_test_acc_cnn_local_epochs_5.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
# plt.show()
plt.close()



plt.figure(figsize=(6, 4))
for (i, color) in enumerate(NUM_TESTS):
    plt.bar(bar_positions.transpose()[i], np.array(bar_values_cnn_10).reshape(5,len(NUM_TESTS)).transpose()[i], color=color, width=BAR_WIDTH, edgecolor='white', label=NAME_TESTS_10_cnn[i])

plt.xlabel('Number of Clients')
plt.ylabel('Communication Rounds')
plt.xticks([bar_positions + BAR_WIDTH for bar_positions in range(len(Grid_Sizes))], ['10', '30', '50', '70', '100'])
#plt.legend(fontsize='small')
plt.ylim([0, 160])
plt.grid(linestyle='-.', linewidth=1,  color="0.9")

plt.savefig("comm_rounds_test_acc_cnn_local_epochs_10.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
# plt.show()
plt.close()

plt.figure(figsize=(6, 4))
for (i, color) in enumerate(NUM_TESTS):
    plt.bar(bar_positions.transpose()[i], np.array(bar_values_nn_5).reshape(5,len(NUM_TESTS)).transpose()[i], color=color, width=BAR_WIDTH, edgecolor='white', label=NAME_TESTS_5_nn[i])

plt.xlabel('Number of Clients')
plt.ylabel('Communication Rounds')
plt.xticks([bar_positions + BAR_WIDTH for bar_positions in range(len(Grid_Sizes))], ['10', '30', '50', '70', '100'])
#plt.legend(fontsize='small')
plt.ylim([0, 300])
plt.grid(linestyle='-.', linewidth=1,  color="0.9")

plt.savefig("comm_rounds_test_acc_nn_local_epochs_5.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
# plt.show()
plt.close()

plt.figure(figsize=(6, 4))
for (i, color) in enumerate(NUM_TESTS):
    plt.bar(bar_positions.transpose()[i], np.array(bar_values_nn_10).reshape(5,len(NUM_TESTS)).transpose()[i], color=color, width=BAR_WIDTH, edgecolor='white', label=NAME_TESTS_10_nn[i])

plt.xlabel('Number of Clients')
plt.ylabel('Communication Rounds')
plt.xticks([bar_positions + BAR_WIDTH for bar_positions in range(len(Grid_Sizes))], ['10', '30', '50', '70', '100'])
#plt.legend(fontsize='small')
plt.ylim([0, 300])
plt.grid(linestyle='-.', linewidth=1, color="0.9")

plt.savefig("comm_rounds_test_acc_nn_local_epochs_10.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
# plt.show()
plt.close()


