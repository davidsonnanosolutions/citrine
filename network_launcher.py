## Test Network
# Sets up and launches the neural net using the provided training data and network2
import magpie_loader_4 as mgpl
import numpy as np
import pandas as pd

training_data, validation_data, test_data = mgpl.load_data_wrapper()

import network2_4 as network2

net = network2.Network([96, 30, 11], cost=network2.QuadraticCost)
net.large_weight_initializer()


k = 0.1
upper = 0.9
step = 0.1
results_path = "/home/spike/citrine/"
"""
while k <= upper:
	ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, 100, 1900, k, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

	epo_df = pd.DataFrame(data=epo)
	epo_df.columns = ['Epoch:']

	cos_df = pd.DataFrame(data=tr_c)
	cos_df.columns = ['Cost:']

	acc_df = pd.DataFrame(data=ev_a)
	acc_df.columns = ['Accuracy:']

	res_df = epo_df.join(cos_df)
	res_df = res_df.join(acc_df)

	file_name = results_path + str(k) + "etas.csv"

	res_df.to_csv(file_name)

	k+=step
"""
ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, 2, 1, 0.2, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

epo_df = pd.DataFrame(data=epo)
epo_df.columns = ['Epoch:']

cos_df = pd.DataFrame(data=tr_c)
cos_df.columns = ['Cost:']

acc_df = pd.DataFrame(data=ev_a)
acc_df.columns = ['Accuracy:']

res_df = epo_df.join(cos_df)
res_df = res_df.join(acc_df)

file_name = results_path + str(k) + "optimized_2000_100_nr.csv"

res_df.to_csv(file_name)


