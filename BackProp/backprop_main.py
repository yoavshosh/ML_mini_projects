import backprop_data
import backprop_network


training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
net = backprop_network.Network([784, 40, 10])

#section_b
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data, save_acc_each_step=True)
net.plot_acc_wrt_epochs()


#section c
training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
net = backprop_network.Network([784, 40, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)
print(f'\nAfter {net.epochs[-1]+1} epochs, we reached an accuracy of {net.test_acc[-1]} on the test data')


#section d
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
net = backprop_network.Network([784, 30, 30, 30, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10000, learning_rate=0.1, test_data=test_data)
net.plot_db_norm_wrt_epochs()