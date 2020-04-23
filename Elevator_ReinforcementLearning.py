import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import gym_lift

import random
import math
import copy

env = gym.make('Lift-v0')
L = env.NUM_LEVEL * (env.NUM_LEVEL+2)
N = env.action_space.n
nl = env.NUM_LEVEL
T_total = env.TOTAL_STEP
print 'Number of floors: ' + str(nl)
print 'Number of actions: ' + str(N)

tune_size = 4
num_neuron = tune_size*N
num_neuron1 = 2*num_neuron + 3*num_neuron/N
num_neuron2 = num_neuron + 2*nl

print 'Number of first hidden layer neurons: ' + str(num_neuron1)

tf.reset_default_graph()

inputs1 = tf.placeholder(shape=[1,L],dtype=tf.float32)

############################ Now declare the weights connecting the input to the hidden layer ##########################

W1 = tf.Variable(tf.random_normal([L, num_neuron1], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([num_neuron1]), name='b1')

################################################ Second layer ##########################################################

W2 = tf.Variable(tf.random_normal([num_neuron1, num_neuron2], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([num_neuron2]), name='b2')

################################ The weights connecting the hidden layer to the output layer ###########################

W3 = tf.Variable(tf.random_normal([N+2*nl, N], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([N]), name='b3')

clip_value_min = -10.0     # to avoid the reward from exploding
clip_value_max = 10.0

################################### Calculate the output of the hidden layer ###########################################

hidden_out1 = tf.add(tf.matmul(inputs1, W1), b1)                                    # weighted sum
hidden_out1_1 = tf.clip_by_value(hidden_out1[0,None,:num_neuron1/2], -1.0, 1.0)     # clip at one for one section
hidden_out1_2 = hidden_out1[0,None,num_neuron1/2:]                                  # 2nd section subject to linear activation
hidden_out1 = tf.concat([hidden_out1_1,hidden_out1_2], axis=-1)                     # combine the two sections

hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
hidden_out2_1 = tf.square(hidden_out2[0,None,:2*nl])                                # take square for one section
hidden_out2_2 = hidden_out2[0,None,2*nl:]
hidden_out2_2 = tf.contrib.layers.maxout(hidden_out2_2,N,axis=-1)                   # maxout the second section
hidden_out2 = tf.concat([hidden_out2_1,hidden_out2_2], axis=-1)                     # combine the two sections

Qout = tf.add(tf.matmul(hidden_out2, W3), b3)                                       # output layer (weighted sum)
print 'Size of output vector: ' + str(Qout.shape)

predict = tf.argmax(Qout,1)

###### Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values #####
nextQ = tf.placeholder(shape=[1,N],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
updateModel = trainer.minimize(loss)
init = tf.global_variables_initializer()

################################################ Set learning parameters ###############################################

y = 0.99                    # discount of future reward
e = 0.1                     # epsilon greedy
num_episodes = 10000        # number of episode to be trained
num_etest = 100             # number of episode for the test

tList = []                  # list of time for training
rList = []                  # list of reward for training
wtList = []                 # list of waiting time for training
ttList = []                 # list of travel time for training
rListT = []                 # list of reward for testing
wtListT = []                # list of waiting time for testing
ttListT = []                # list of travel time for testing
m = 200.0                   # number of data points for the plotting of the learning process (for figure 1)
M = int(m)

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):

        s = env.reset()                        # reset environment and get first new observation
        rAll = 0
        d = False
        j = 0                                  # step
        t = 0.0                                # time
        Sum_fR = 0.0                           # number of step which considers future reward (i.e. not terminal state)

        e = 1.0 - (1.0/num_episodes)*i         # epsilon greedy

        while t < T_total:                       # reinforcement learning
            j += 1

            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s.reshape((1,L))})        # choose an action by greedily (with e chance of random action) from the Q-network

            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            s1,r,d,fR = env.step(a[0])                                  # get new state and reward from environment
            Sum_fR += fR

            Q1 = sess.run(Qout,feed_dict={inputs1:s1.reshape((1,L))})   # obtain the Q' values by feeding the new state through our network

            maxQ1 = np.max(Q1)                                          # obtain maxQ' and set our target value for chosen action.
            targetQ = allQ

            if fR > 0.0:
                targetQ[0,a[0]] = r + fR*y*maxQ1                        # targetQ is a (1,N) array, change value of one of the element
            else:
                targetQ[0,a[0]] = r                                     # terminal state, do not consider future reward
                                                                        # error back propagation
            _, weight1, bias1, weight2, bias2, weight3, bias3 = sess.run([updateModel, W1, b1, W2, b2, W3, b3], feed_dict={inputs1:s.reshape((1,L)), nextQ:targetQ})

            rAll += r                                                   # accumulated reward
            s = s1                                                      # update state
            t = env.get_time()                                          # get current time

            if d == True:
                break

        tList.append(t)
        rList.append(rAll)
        wt = env.get_waiting_time()                                     # get the list of waiting time for this episode
        wtList.append(np.mean(wt))                                      # mean waiting time
        tt = env.get_travel_time()                                      # get the list of travel time for this episode
        ttList.append(np.mean(tt))                                      # mean travel time

        if i % 100 == 0:
            print 'Episode: ' + str(i)
            print 'Total number of step: ' + str(j)
            print 'Total number of non-terminal step: ' + str(Sum_fR)
            print 'Mean waiting time (over all trained episodes): ' + str(np.mean(wtList))

    #################################################### Testing #######################################################

    for ii in range(num_etest):
        s = env.reset()                        # reset environment and get first new observation
        rAll = 0
        d = False
        j = 0
        t = 0.0
        while t < T_total:
            j += 1
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:s.reshape((1,L))})
            s1,r,d,_ = env.step(a[0])
            rAll += r
            s = s1
            t = env.get_time()
            if d == True:
                break

        if ii == 0:
            states_episode = env.get_state_dynamics()       # get the state at each step
            actions_episode = env.get_action_dynamics()     # get the list of actions
            time_episode = env.get_time_dynamics()          # get the list of time

        rListT.append(rAll)
        wt = env.get_waiting_time()
        wtListT.append(np.mean(wt))
        tt = env.get_travel_time()
        ttListT.append(np.mean(tt))
        #print 'Number of commuter picked up:' +str(len(wt))

#################################### Plot the mean waiting time over the training process ##############################

time = np.arange(m/2, num_episodes, m)
Twaiting = np.zeros(len(time))
Ttravelling = np.zeros(len(time))
for t in range(len(time)):
    wt = wtList[t*M:(t+1)*M]                  # get average waiting time per episode by period
    Twaiting[t] = sum(wt)/m                   # waiting per episode averaged over m episodes
    tt = ttList[t*M:(t+1)*M]
    Ttravelling[t] = sum(tt)/m

fig1 = plt.figure(num=1, figsize=(5, 5.5), dpi=100, facecolor='w', edgecolor='k')
ax1 = fig1.add_subplot(211)
plt.plot(time, Twaiting, 'o-', markersize=4)
plt.xlabel('Episode')
plt.ylabel('Waiting time')
ax2 = fig1.add_subplot(212)
plt.plot(time, Ttravelling, 'o-', markersize=4)
plt.xlabel('Episode')
plt.ylabel('Travelling time')
plt.subplots_adjust(top=0.93, bottom=0.12, left=0.16, right=0.92, hspace=0.28, wspace=0.28)
filename1 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_training.pdf'
plt.savefig(filename1, format='pdf', dpi=300)

############################################### Save the trained weight ################################################

filename2 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_weight1.npy'
filename3 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_weight2.npy'
filename4 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_weight3.npy'
filename5 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_bias1.npy'
filename6 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_bias2.npy'
filename7 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_bias3.npy'

np.save(filename2, weight1)
np.save(filename3, weight2)
np.save(filename4, weight3)
np.save(filename5, bias1)
np.save(filename6, bias2)
np.save(filename7, bias3)

##################################### Save the waiting and travelling time for the test ################################

filename1 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_waittime.txt'
np.savetxt(filename1, wtListT, fmt='%f')
filename2 = 'OptimizeTime_Level' + str(nl) + '_neuron' + str(num_neuron1) + '_traveltime.txt'
np.savetxt(filename2, ttListT, fmt='%f')

print 'Mean waiting time (test): ' + str(np.mean(wtListT))
print 'Mean travelling time (test): ' + str(np.mean(ttListT))

plt.show()



