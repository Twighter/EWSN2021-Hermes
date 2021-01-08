import numpy as np
import pandas as pd
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            network_idx,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy_1=0.95,
            e_greedy_2=0.95,
            replace_target_iter=200,
            memory_size=2000,
            batch_size=32,
            output_graph = False,
            randSeed = 0
    ):
        np.random.seed(randSeed)
        tf.set_random_seed(randSeed)

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.epsilon_1 = e_greedy_1
        self.epsilon_2 = e_greedy_2
        self.preNum = 10
        self.preAction = np.random.randint(0, self.n_actions - 1, self.preNum)
        for idx in range(self.preNum // 2):
            self.preAction[idx] = self.n_actions - 1

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.network_idx = network_idx
        self.network_name = 'Net_' + str(self.network_idx)

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(self.network_name + 'target_net_params')
        e_params = tf.get_collection(self.network_name + 'eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        with tf.variable_scope(self.network_name):
            # ------------------ build evaluate_net ------------------
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)
            n_l1 = 10 #(self.n_features + self.n_actions) // 2 + 3
            with tf.variable_scope('eval_net'):
                c_names = [self.network_name + 'eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('l1'):
                    w1 = tf.get_variable("w1", [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable("b1", [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable("w2", [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable("b2", [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            # ------------------ build target_net ------------------
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
            with tf.variable_scope('target_net'):
                c_names = [self.network_name + 'target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('l1'):
                    w1 = tf.get_variable("w1", [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable("b1", [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

                with tf.variable_scope('l2'):
                    w2 = tf.get_variable("w2", [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable("b2", [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

            self.paramNum = 8

    def clear_transition(self):
        self.memory_counter = 0

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        randomTemp = np.random.uniform()
        if randomTemp < self.epsilon_1:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        elif randomTemp < self.epsilon_2:
            action = self.preAction[np.random.randint(0, self.preNum)].item()
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.learn_step_counter += 1

    def loadModel(self):
        loadModelEval = tf.get_collection(self.network_name + 'eval_net_params')
        loadModelTarget = tf.get_collection(self.network_name + 'target_net_params')
        return loadModelEval, loadModelTarget

    def saveModel(self, loadModelEval, loadModelTarget):
        saveModelEval = tf.get_collection(self.network_name + 'eval_net_params')
        saveModelTarget = tf.get_collection(self.network_name + 'target_net_params')
        swapModelEvalOp = [tf.assign(S, L) for S, L in zip(saveModelEval, loadModelEval)]
        swapModelTargetOp = [tf.assign(S, L) for S, L in zip(saveModelTarget, loadModelTarget)]

        self.sess.run(swapModelEvalOp)
        self.sess.run(swapModelTargetOp)

    def getParameter(self):
        networkOffset = self.network_idx * self.paramNum
        param = tf.trainable_variables()

        paramSet = []
        for paramIdx in range(self.paramNum):
            paramSet.append(self.sess.run(param[networkOffset + paramIdx]))

        return paramSet
