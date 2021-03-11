import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics

def calculate_average(values):
    # print(values)
    return np.mean(values)

def calulte_standard_dev(values):
    
    # for i in range(len(list_values)):
    #     std_values.append(statistics.stdev(list_values[i]))
    # return std_values

    return statistics.stdev(values)


iter_list = []
acc_list = []
time_list = []
invoke_time_list = []
agg_time_list = []
train_time_list = []

iter_list_10 = []
acc_list_10 = []
time_list_10 = []
invoke_time_list_10 = []
agg_time_list_10 = []

iter_list_nn_10 = []
acc_list_nn_10 = []
time_list_nn_10 = []
invoke_time_list_nn_10 = []
agg_time_list_nn_10 = []

iter_list_nn_5 = []
acc_list_nn_5 = []
time_list_nn_5 = []
invoke_time_list_nn_5 = []
agg_time_list_nn_5 = []
train_time_list_nn_5 = []

cold_start_cnn_invoke_time = []
cold_start_cnn_agg_time = []
cold_start_cnn_training_time = []
cold_start_cnn_total_time = []

cold_start_nn_invoke_time = []
cold_start_nn_agg_time = []
cold_start_nn_training_time = []
cold_start_nn_total_time = []



# with open('log_cnn_5_acc_time_iter.log') as file:
#     lines = file.readlines()
#     lines = [line.rstrip() for line in lines]
#     new_line = [line.split(",") for line in lines]
with open('cold_star_cnn_5.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_cold_start = [line.split(",") for line in lines]

with open('cold_start_nn_5.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_cold_start_nn = [line.split(",") for line in lines]

with open('log_cnn_5_acc_time_iter_new.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line = [line.split(",") for line in lines]

with open('log_cnn_10_acc_time_iter.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_10 = [line.split(",") for line in lines]

with open('log_nn_10_acc_iter.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_nn_10 =  [line.split(",") for line in lines]

with open('log_nn_5_acc_iter.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_nn_5 =  [line.split(",") for line in lines]

# with open('log_nn_5_acc_time_iter.log') as file:
#     lines = file.readlines()
#     lines = [line.rstrip() for line in lines]
#     new_line_nn_5_time =  [line.split(",") for line in lines]


with open('log_test_mnist_nn_100_clients_adam_5_local_epochs_fixed_final_training_time.log') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    new_line_nn_5_time =  [line.split(",") for line in lines]


for i in range(0, len(new_line_cold_start_nn)):
    line = new_line_cold_start_nn[i]
    line = [line.strip() for line in line]
    invoker_time = float(line[2].split(":")[1].strip())
    agg_time = float(line[3].split(":")[1].strip())
    time = float(line[4].split(":")[1].strip())
    train_time = float(line[5].split(":")[1].strip())

    cold_start_nn_invoke_time.append(invoker_time)
    cold_start_nn_agg_time.append(agg_time - 5)
    cold_start_nn_training_time.append(train_time - 52)
    cold_start_nn_total_time.append(time)

for i in range(0, len(new_line_cold_start)):
    line = new_line_cold_start[i]
    line = [line.strip() for line in line]
    invoker_time = float(line[2].split(":")[1].strip())
    agg_time = float(line[3].split(":")[1].strip())
    time = float(line[4].split(":")[1].strip())
    train_time = float(line[5].split(":")[1].strip())

    cold_start_cnn_invoke_time.append(invoker_time)
    cold_start_cnn_agg_time.append(agg_time - 5)
    cold_start_cnn_training_time.append(train_time - 52)
    cold_start_cnn_total_time.append(time)

# print(cold_start_cnn_invoke_time)
# print(cold_start_cnn_agg_time)
# print(cold_start_cnn_training_time)
# print(cold_start_cnn_total_time)


for i in range(0, len(new_line)):
    # print(new_line[i])
    # line = new_line[i][:3]
    line = new_line[i][:6]
    line = [line.strip() for line in line]
    # print(line)
    # print(line)
    iter_num = int(line[0].split(":")[1])
    acc = float(line[1].split(":")[1].strip())
    invoker_time = float(line[2].split(":")[1].strip())
    agg_time = float(line[3].split(":")[1].strip())
    time = float(line[4].split(":")[1].strip())
    train_time = float(line[5].split(":")[1].strip())
    iter_list.append(iter_num)
    acc_list.append(acc)
    time_list.append(time)
    invoke_time_list.append(invoker_time)
    agg_time_list.append(agg_time)
    train_time_list.append(train_time)

    # print(acc)
    # print(iter_num)
    # print(agg_time)
    # print(time)
    # print(invoker_time)
# print(new_line_nn_10)

# print(train_time_list)

for i in range(0, len(new_line_10)):
    line = new_line_10[i][:6]
    line = [line.strip() for line in line]
    iter_num = int(line[0].split(":")[1]) + 1
    acc = float(line[4].split(":")[1].strip())
    invoker_time = float(line[2].split(":")[1].strip())
    agg_time = float(line[3].split(":")[1].strip())
    time = float(line[5].split(":")[1].strip())

    
    iter_list_10.append(iter_num)
    acc_list_10.append(acc)
    time_list_10.append(time)
    invoke_time_list_10.append(invoker_time)
    agg_time_list_10.append(agg_time)


for i in range(0, len(new_line_nn_10)):
    line = new_line_nn_10[i][:6]
    line = [line.strip() for line in line]
    iter_num = int(line[0].split(":")[1]) + 1
    acc = float(line[4].split(":")[1].strip())
    invoker_time = float(line[2].split(":")[1].strip())
    agg_time = float(line[3].split(":")[1].strip())
    time = float(line[5].split(":")[1].strip())

    iter_list_nn_10.append(iter_num)
    acc_list_nn_10.append(acc)
    time_list_nn_10.append(time)
    invoke_time_list_nn_10.append(invoker_time)
    agg_time_list_nn_10.append(agg_time)
    # print(line)

for i in range(0, len(new_line_nn_5)):
    line = new_line_nn_5[i]
    line = [line.strip() for line in line]
    # print(line)
    iter_num = int(line[0].split(":")[1]) + 1
    acc = float(line[2].split(":")[1].strip())
    iter_list_nn_5.append(iter_num)
    acc_list_nn_5.append(acc)

for i in range(0, len(new_line_nn_5_time)):
    line = new_line_nn_5_time[i]
    # print(line)
    line = [line.strip() for line in line]

    time = float(line[5].split(":")[1].strip())
    invoker_time = float(line[3].split(":")[1].strip())
    agg_time = float(line[4].split(":")[1].strip())
    train_time = float(line[6].split(":")[1].strip())

    time_list_nn_5.append(time)
    invoke_time_list_nn_5.append(invoker_time)
    agg_time_list_nn_5.append(agg_time)
    train_time_list_nn_5.append(train_time)


    # print(line)





font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 16}
matplotlib.rcParams.update({'font.size': 14})

plt.figure(figsize=(6,4))
plt.plot(iter_list, time_list, "*:", color='tab:blue', label=("local-epochs=5, lr=1e-3"), linewidth=1.5)
plt.plot(iter_list_10, time_list_10, "x--", color='tab:orange', label=("local-epochs=10, lr=1e-3"), linewidth=1.5)
plt.ylim([0, 165])
plt.xlabel('Communication Rounds')
plt.ylabel('Time (sec)')
#plt.legend(fontsize='small')
# plt.show()
plt.savefig("Mnist_cnn_time_per_iteration_num_epochs.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)


plt.close()

plt.figure(figsize=(6,4))
plt.plot(iter_list, acc_list, "*:", color='tab:blue', label=("local-epochs=5"), linewidth=1.5)
plt.plot(iter_list_10, acc_list_10, "x--", color='tab:orange', label=("local-epochs=10"), linewidth=1.5)
plt.ylim([0.4, 1])
plt.axvline(x=13, linestyle='--', color='tab:red')
plt.axhline(y=0.95,  linestyle='--', color='tab:red')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Set Accuracy')
#plt.legend(fontsize='small')
plt.savefig("Mnist_cnn_acc_rounds_num_epochs.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)

plt.close()

plt.figure(figsize=(6,4))
plt.plot(iter_list_nn_5, time_list_nn_5, "*:", color='tab:blue', label=("local-epochs=5"), linewidth=1.5)
plt.plot(iter_list_nn_10, time_list_nn_10, "x--", color='tab:orange', label=("local-epochs=10"), linewidth=1.5)
plt.ylim([0, 140])
plt.xlabel('Communication Rounds')
plt.ylabel('Time (sec)')
#plt.legend(fontsize='small')
# plt.show()
plt.savefig("Mnist_nn_time_per_iteration_num_epochs.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
plt.close()


plt.figure(figsize=(6,4))
plt.plot(iter_list_nn_5, acc_list_nn_5, "*:", color='tab:blue', label=("local-epochs=5"), linewidth=1.5)
plt.plot(iter_list_nn_10, acc_list_nn_10, "x--", color='tab:orange', label=("local-epochs=10"), linewidth=1.5)
plt.ylim([0.4, 1])
plt.axvline(x=31, linestyle='--', color='tab:red')
plt.axvline(x=91, linestyle='--', color='tab:red')
plt.axhline(y=0.95,  linestyle='--', color='tab:red')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Set Accuracy')
#plt.legend(fontsize='small')
# plt.show()
plt.savefig("Mnist_nn_acc_rounds_num_epochs.png", dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
plt.close()

line_num_skip = 1
training_time_average = calculate_average(train_time_list[line_num_skip:])
training_time_std = calulte_standard_dev(train_time_list[line_num_skip:])
invoke_time_average = calculate_average(invoke_time_list[line_num_skip:])
invoke_time_std = calulte_standard_dev(invoke_time_list[line_num_skip:])
agg_time_average = calculate_average(agg_time_list[line_num_skip:])
agg_time_std = calulte_standard_dev(agg_time_list[line_num_skip:])
total_time_average = calculate_average(time_list[line_num_skip:])
total_time_std = calulte_standard_dev(time_list[line_num_skip:])


# print(training_time_std)
# print(invoke_time_average)
# print(invoke_time_std)
# print(agg_time_average)
# print(agg_time_std)
# print(total_time_average)
# print(total_time_std)
misc_time = total_time_average - training_time_average - invoke_time_average - agg_time_average
# print(misc_time)




cold_start_cnn_time_average =  calculate_average(cold_start_cnn_total_time)
cold_start_cnn_time_std = calulte_standard_dev(cold_start_cnn_total_time)
cold_start_cnn_invoketime_average = calculate_average(cold_start_cnn_invoke_time)
cold_start_cnn_invoketime_std = calulte_standard_dev(cold_start_cnn_invoke_time)
cold_start_cnn_aggtime_average = calculate_average(cold_start_cnn_agg_time)
cold_start_cnn_aggtime_std = calulte_standard_dev(cold_start_cnn_agg_time)
cold_start_cnn_traintime_average = calculate_average(cold_start_cnn_training_time)
cold_start_cnn_traintime_std = calulte_standard_dev(cold_start_cnn_training_time)



cold_start_nn_time_average =  calculate_average(cold_start_nn_total_time)
cold_start_nn_time_std = calulte_standard_dev(cold_start_nn_total_time)
cold_start_nn_invoketime_average = calculate_average(cold_start_nn_invoke_time)
cold_start_nn_invoketime_std = calulte_standard_dev(cold_start_nn_invoke_time)
cold_start_nn_aggtime_average = calculate_average(cold_start_nn_agg_time)
cold_start_nn_aggtime_std = calulte_standard_dev(cold_start_nn_agg_time)
cold_start_nn_traintime_average = calculate_average(cold_start_nn_training_time)
cold_start_nn_traintime_std = calulte_standard_dev(cold_start_nn_training_time)

# print(cold_start_cnn_time_average)
# print(cold_start_nn_time_average)
# print(cold_start_nn_time_std)
# print(cold_start_nn_invoketime_average)
# print(cold_start_nn_invoketime_std)
# print(cold_start_nn_aggtime_average)
# print(cold_start_nn_aggtime_std)
print(training_time_average)
print(cold_start_nn_traintime_average)
print(cold_start_cnn_traintime_average)

# print(cold_start_nn_traintime_std)


# print(cold_start_cnn_time_average)
# print(cold_start_cnn_time_std)

# print(cold_start_cnn_invoketime_average)
# print(cold_start_cnn_invoketime_std)

# print(cold_start_cnn_aggtime_average)
# print(cold_start_cnn_aggtime_std)


# print(cold_start_cnn_traintime_std)

# print(time_list_nn_5)
# print(invoke_time_list_nn_5)
# print(agg_time_list_nn_5)
# print(train_time_list_nn_5)

training_time_average_nn = calculate_average(train_time_list_nn_5[line_num_skip:])
training_time_std_nn = calulte_standard_dev(train_time_list_nn_5[line_num_skip:])
invoke_time_average_nn = calculate_average(invoke_time_list_nn_5[line_num_skip:])
invoke_time_std_nn = calulte_standard_dev(invoke_time_list_nn_5[line_num_skip:])
agg_time_average_nn = calculate_average(agg_time_list_nn_5[line_num_skip:])
agg_time_std_nn = calulte_standard_dev(agg_time_list_nn_5[line_num_skip:])
total_time_average_nn = calculate_average(time_list_nn_5[line_num_skip:])
total_time_std_nn = calulte_standard_dev(time_list_nn_5[line_num_skip:])
print(training_time_average_nn)

# print(training_time_std_nn)
# print(invoke_time_average_nn)
# print(invoke_time_std_nn)
# print(agg_time_average_nn)
# print()
# print(agg_time_std)
# print(cold_start_nn_aggtime_std)
# print(agg_time_std_nn)
# print(total_time_average_nn)
# print(total_time_std_nn)



N = 4
ind = [0, 0.2, 0.4, 0.6]
width = 0.1
# print(ind)


invoke_time_averages = [cold_start_cnn_invoketime_average, invoke_time_average, cold_start_nn_invoketime_average, invoke_time_average_nn]
invoke_time_stds = [cold_start_cnn_invoketime_std, invoke_time_std, cold_start_nn_invoketime_std, invoke_time_std_nn]

all_training_time_averages = [cold_start_cnn_traintime_average, training_time_average, cold_start_nn_traintime_average, training_time_average_nn]

training_time_averages = [cold_start_cnn_traintime_average, cold_start_nn_traintime_average]

training_time_averages_without_cold = [training_time_average, training_time_average_nn]

training_time_stds = [cold_start_cnn_traintime_std, training_time_std, cold_start_nn_traintime_std, training_time_std_nn]
aggregate_time_averages = [cold_start_cnn_aggtime_average, agg_time_average, cold_start_nn_aggtime_average, agg_time_average_nn]
aggregate_time_stds = [cold_start_cnn_aggtime_std, agg_time_std, cold_start_nn_aggtime_std, agg_time_std_nn]
invoker_start_time = [52, 0, 52, 0]
agg_start_time = [5, 0, 5, 0]

new_list = [sum(x) for x in zip(invoke_time_averages, invoker_start_time, all_training_time_averages, agg_start_time)]

second_layer_sum_1 = [sum(x) for x in zip(invoke_time_averages, invoker_start_time)]

second_layer_sum_1 = [second_layer_sum_1[0], second_layer_sum_1[2]]
second_layer_sum_2 = [sum(x) for x in zip(invoke_time_averages, invoker_start_time)]
second_layer_sum_2 = [second_layer_sum_2[1], second_layer_sum_2[3]]

second_layer_sum = [sum(x) for x in zip(invoke_time_averages, invoker_start_time)]

third_layer_sum = [sum(x) for x in zip(second_layer_sum, all_training_time_averages)]

print(new_list)
# p1 = plt.bar(ind, invoke_time_average, width=0.8, yerr=invoke_time_std, capsize=3, color="tab:orange")
# p2 = plt.bar(ind, training_time_average, width=0.8, yerr=training_time_std, capsize=3, color="tab:blue")
# p3 = plt.bar(ind, agg_time_average, width=0.8, yerr=agg_time_std, capsize=3, color="tab:green")

print(training_time_averages)
plt.figure(figsize=(6.5, 5))
#matplotlib.rcParams.update({'font.size': 3})
matplotlib.rcParams['hatch.linewidth'] = 1 # previous svg hatch linewidth
# p1 = plt.bar(ind, invoke_time_average, width=0.8 , yerr=invoke_time_std, capsize=2, color="tab:orange")
p1 = plt.bar(ind, invoke_time_averages, width=width , yerr=invoke_time_stds , capsize=2, ecolor='blue', edgecolor='black', color = 'w', hatch = 'oo' )
p2 = plt.bar(ind, invoker_start_time,  width=width,  yerr=[10, 0, 10, 0], bottom=invoke_time_averages, ecolor='blue',capsize=2, edgecolor='black', color = 'w', hatch = '////')
p3 = plt.bar([ind[0], ind[2]], training_time_averages,  width=width, yerr=[training_time_stds[0],training_time_stds[2]] , bottom=second_layer_sum_1, ecolor='blue',capsize=2, edgecolor='black', color = 'w', hatch = '\\\\')
p4 = plt.bar([ind[1], ind[3]], training_time_averages_without_cold,  width=width, yerr=[training_time_stds[1],training_time_stds[3]], bottom=second_layer_sum_2, ecolor='blue',capsize=2, edgecolor='black', color = 'w', hatch = '+++')

p5 = plt.bar(ind, agg_start_time,  width=width, bottom=third_layer_sum, capsize=2, edgecolor='black', ecolor='blue',color = 'w', hatch = '**')
# p2 = plt.bar(ind, agg_time_average, bottom=invoke_time_average, width=0.8, color="tab:green", yerr=agg_time_std, capsize=2)
# p3 = plt.bar(ind, training_time_average, bottom=agg_time_average, width=0.8, yerr=training_time_std, capsize=2, color="tab:blue")
p6 = plt.bar(ind, aggregate_time_averages, bottom= new_list, width=width, yerr=aggregate_time_stds, capsize=2, ecolor='blue', edgecolor='black', color = 'w', hatch="xxx")





# p3 = plt.bar(ind, agg_time_average, width=0.8, color="tab:green")

# plt.xlabel('Communication Rounds')

plt.ylabel('Time (sec)')
plt.xticks(ind, ['CNN (cold start)', 'CNN', 'NN (cold start)', 'NN'])
plt.ylim([0, 160])
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('Invocation time', "Invokers start time", 'Clients start + training time',
                                                         'Training time',
                                          'Aggregator start time', 'Aggregation time',
                                                ), fontsize='7', loc='upper right')

#plt.grid(axis="both", color="0.9", linestyle='-', linewidth=1)
#plt.xticks([])
#plt.set_xticks(range(0,len(dataset.index)))

#axs.set_axisbelow(True)
plt.savefig("time_separation.png", format='png', dpi=300, bbox_inches='tight')

# N = 2
# ind = np.arange(N) 
# width = 0.2
# # print(ind)
# plt.figure(figsize=(6, 4))

# invoke_time_cold_averages = [cold_start_cnn_invoketime_average, cold_start_nn_invoketime_average]
# invoke_time_cold_stds = [cold_start_cnn_invoketime_std, cold_start_nn_invoketime_std]
# training_time_cold_averages = [cold_start_cnn_traintime_average, cold_start_nn_traintime_average]
# training_time_cold_stds = [cold_start_cnn_traintime_std, cold_start_nn_traintime_std]
# aggregate_time_cold_averages = [cold_start_cnn_aggtime_average, cold_start_nn_aggtime_average]
# aggregate_time_cold_stds = [cold_start_cnn_aggtime_std, cold_start_nn_aggtime_std]

# new_list_cold = [sum(x) for x in zip(invoke_time_cold_averages, training_time_cold_averages)]
# # print(new_list)
# # p1 = plt.bar(ind, invoke_time_average, width=0.8, yerr=invoke_time_std, capsize=3, color="tab:orange")
# # p2 = plt.bar(ind, training_time_average, width=0.8, yerr=training_time_std, capsize=3, color="tab:blue")
# # p3 = plt.bar(ind, agg_time_average, width=0.8, yerr=agg_time_std, capsize=3, color="tab:green")

# # p1 = plt.bar(ind, invoke_time_average, width=0.8 , yerr=invoke_time_std, capsize=2, color="tab:orange")
# p1 = plt.bar(ind, invoke_time_cold_averages, width=width , yerr=invoke_time_cold_stds , capsize=2, color="lightsteelblue")
# p2 = plt.bar(ind, training_time_cold_averages,  width=width, yerr=training_time_cold_stds, bottom=invoke_time_cold_averages, capsize=2, color="cornflowerblue")
# # p2 = plt.bar(ind, agg_time_average, bottom=invoke_time_average, width=0.8, color="tab:green", yerr=agg_time_std, capsize=2)
# # p3 = plt.bar(ind, training_time_average, bottom=agg_time_average, width=0.8, yerr=training_time_std, capsize=2, color="tab:blue")
# p3 = plt.bar(ind, aggregate_time_cold_averages, bottom= new_list_cold, width=width, yerr=aggregate_time_cold_stds, capsize=2, color="royalblue") 

# # p3 = plt.bar(ind, agg_time_average, width=0.8, color="tab:green")

# # plt.xlabel('Communication Rounds')
# plt.ylabel('Time (sec)')
# plt.xticks(ind, ['CNN', 'NN'])
# plt.ylim([0, 160])
# plt.legend((p1[0], p2[0], p3[0]), ('Invocation Time', 'Training Time', 'Aggregation Time'), fontsize='small')
# plt.show()