'''前言,先建立一个简单的HMM模型'''

import numpy as np


class HMM():
    def __init__(self, n_state, n_observation):
        self.n_state = n_state  # 隐状态的个数
        self.n_observation = n_observation  # 观测状态的个数

        # 初始化模型参数
        self.initial_prob = np.ones(n_state) / n_state  # 初始概率向量
        self.transition_prob = np.ones((n_state, n_state)) / n_state  # 转移概率矩阵
        self.emission_prob = np.ones((n_state, n_observation)) / n_observation  # 发射概率矩阵

    def train(self, observations, iterations=100):
        # Baum-Welch算法，用于训练HMM模型参数

        for _ in range(iterations):
            # 初始化变量用于累计更新模型参数
            new_initial_prob = np.zeros(self.n_state)
            new_transition_prob = np.zeros((self.n_state, self.n_state))
            new_emission_prob = np.zeros((self.n_state, self.n_observation))

            for observation in observations:
                # 将观测状态映射到合法范围内
                observation = np.clip(observation, 0, self.n_observation - 1)

                # 前向算法
                alpha = np.zeros((len(observation), self.n_state))
                alpha[0] = self.initial_prob * self.emission_prob[:, observation[0]]
                for t in range(1, len(observation)):
                    alpha[t] = np.dot(alpha[t - 1], self.transition_prob) * self.emission_prob[:, observation[t]]

                # 后向算法
                beta = np.zeros((len(observation), self.n_state))
                beta[-1] = 1
                for t in range(len(observation) - 2, -1, -1):
                    beta[t] = np.dot(self.transition_prob, self.emission_prob[:, observation[t + 1]] * beta[t + 1])

                # 更新模型参数
                new_initial_prob += alpha[0]
                for t in range(len(observation) - 1):
                    new_transition_prob += (alpha[t][:, np.newaxis] * self.transition_prob *
                                            self.emission_prob[:, observation[t + 1]] * beta[t + 1])
                for t in range(len(observation)):
                    new_emission_prob[:, observation[t]] += alpha[t] * beta[t]

            # 归一化模型参数
            self.initial_prob = new_initial_prob / np.sum(new_initial_prob)
            self.transition_prob = new_transition_prob / np.sum(new_transition_prob, axis=1)[:, np.newaxis]
            self.emission_prob = new_emission_prob / np.sum(new_emission_prob, axis=1)[:, np.newaxis]

    def predict(self, observation):
        # 维特比算法，用于预测给定观测序列下的最可能的隐状态序列

        # 初始化变量
        T = len(observation)
        delta = np.zeros((T, self.n_state))
        psi = np.zeros((T, self.n_state), dtype=int)

        # 初始化初始状态
        delta[0] = self.initial_prob * self.emission_prob[:, observation[0]]

        # 递推计算最大概率路径
        for t in range(1, T):
            for j in range(self.n_state):
                delta[t, j] = np.max(delta[t - 1] * self.transition_prob[:, j] * self.emission_prob[j, observation[t]])
                psi[t, j] = np.argmax(delta[t - 1] * self.transition_prob[:, j])

        # 回溯得到最可能的隐状态序列
        states = [np.argmax(delta[-1])]
        for t in range(T - 1, 0, -1):
            states.append(psi[t, states[-1]])

        return list(reversed(states))