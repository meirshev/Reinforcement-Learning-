from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import math

class DeepMindPolicy(bp.Policy):

    trainingData = {}
    batchSize = 10
    init_needed = True
    gamma = 0.9
    radius = 9
    learningRate = 0.01
    actions_to_ints = {'CC': 0, 'CW': 1, 'CN': 2}

    def cast_string_args(self, policy_args):

        policy_args['useless'] = int(policy_args['useless'])
        return policy_args
        # self.learningRate = int(policy_args['learningRate'])
        # self.gamma = int(policy_args['gamma'])
        # self.radius = int(policy_args['radius'])

    def init_run(self):

        if self.init_needed:

            sizeOfImg = np.power((self.radius * 2) + 1, 2) + 1

            self.x = tf.placeholder(tf.float32, [None, sizeOfImg])
            self.W_1 = tf.Variable(tf.random_normal([sizeOfImg, sizeOfImg]))
            self.b_1 = tf.Variable(tf.random_normal([sizeOfImg]))
            self.hidden_1 = tf.nn.relu(tf.matmul(self.x, self.W_1) + self.b_1)

            self.W_2 = tf.Variable(tf.random_normal([sizeOfImg, sizeOfImg]))
            self.b_2 = tf.Variable(tf.random_normal([sizeOfImg]))
            self.hidden_2 = tf.nn.relu(tf.matmul(self.hidden_1, self.W_2) + self.b_2)

            self.W_3 = tf.Variable(tf.random_normal([sizeOfImg, 3]))
            self.b_3 = tf.Variable(tf.random_normal([3]))
            self.y = tf.matmul(self.hidden_2, self.W_3) + self.b_3


            self.y_ = tf.placeholder(tf.float32, [None, 3])
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.q_a))
            self.train_step = tf.train.AdamOptimizer(self.learningRate).minimize(self.cross_entropy)

            self.sess = tf.InteractiveSession() # maybe it should be non-interActiveSession
            tf.global_variables_initializer().run()

            self.init_needed = False
            # self.correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def learn(self, reward, t):

        if t % 20 == 0:

            data_dic = {'currentStates': np.array([]), 'nextState': np.array([]),
                        'rewards': np.array([]), 'actions': np.array([])}

            # Generate batch
            batch = np.array(np.random.choice(list(self.trainingData.values()),
                                              self.batchSize, replace=False))

            # Extract data from batch
            data_dic['currentState'] = batch[:, 0]
            data_dic['actions'] = batch[:, 1]
            data_dic['rewards'] = batch[:, 2]
            data_dic['nextState'] = batch[:, 3]

            # Convert actions into integers
            actions = [self.actions_to_ints[action] for action in data_dic['actions']]

            # Calculate max(Q(s',a)) with respect to a.
            temp = self.sess.run(self.y, feed_dict={self.x: data_dic['nextState']})
            qNext = np.array([np.max(row) for row in temp])

            prediction = np.array(data_dic['rewards']) + self.gamma * qNext
            Q_nxtState = self.sess.run(self.y, feed_dict={self.x: data_dic['currentState']})
            Q_nxtState[np.arange(self.batchSize), np.array(actions)] = prediction

            # Update Q.
            self.sess.run(self.train_step, feed_dict={self.x: data_dic['currentState'],
                                                      self.y_: Q_nxtState})
        # Store new data
        self.trainingData[t] = np.array([None, None, reward, None])

    def act(self, t, state, player_state):

        head_pos = player_state['chain'][-1]
        start_pos = [head_pos[0] - self.radius, head_pos[1] - self.radius]
        s = (self.radius * 2 + 1, self.radius * 2 + 1)
        relevantPart = np.zeros(s, dtype=np.int)

        for i in range(start_pos[0], start_pos[0] + (2 * self.radius) + 1):
            for j in range(start_pos[1], start_pos[1] + (2 * self.radius) + 1):
                relevantPart[i-start_pos[0]][j- start_pos[1]] = state[i % state.shape[0]][j % state.shape[1]]
        relevantPart = relevantPart.reshape((self.radius * 2) + 1)
        netInput = np.append(relevantPart, [player_state[1]])

        # Generate actions.
        randomAction = np.random.randint(3)
        netPrediction = np.max(self.sess.run(self.y, feed_dict={self.x: netInput}))

        # 10% chance for random action.
        action = bp.Policy.ACTIONS[np.random.choice([randomAction, netPrediction], p=[0.1, 0.9])]

        # Storing remaining data
        if t is not 0:
            self.trainingData[t-1][3] = relevantPart
        self.trainingData[t][0] = relevantPart
        self.trainingData[t][1] = action

        return action

    def get_state(self):
        pass




