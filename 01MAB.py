# Multi armed Bandit（MAB）即多臂老虎机问题
# 目标是在操作T次拉杆后获得尽可能高的累计奖励。
# 由于奖励的概率分布是未知的，因此我们需要在
# "探索拉杆的获奖概率"和"根据经验选择获奖最多的拉杆"
# 中进行权衡。"采用怎样的操作策略才能使获得的累积奖
# 励最高"便是多臂老虎机问题。

# MAB问题的目标为最大化累计奖励，等价于最小化累计懊悔

""" 1. 拉杆数为10的伯努利多臂老虎机，奖励服从伯努利分布 """

# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt

class BernouliBandit:
    """ 伯努利多臂老虎机，输入k表示拉杆个数"""
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0~1的数，作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs) # 获取概率最大的拉杆
        self.best_prob = self.probs[self.best_idx] # 最大的获奖概率
        self.k = K
    
    def step(self, k):
        # 当玩家选择了k号拉杆后，根据拉动该老虎机的k号拉杆获得奖励的概率返回
        # 1（获奖）或 0（未获奖）

        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
np.random.seed(1) # 设定随机种子，使实验具有可重复性
K = 10
bandit_10_arm = BernouliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

"""Solver类实现上述多臂老虎机的求解方案"""
# 根据策略选择动作、根据动作获取奖励、更新期望奖励估值、更新累计懊悔和计数
# 将前三者放在run_one_step()函数中，由每个继承类的策略具体实现。
# 而更新累积懊悔和计数则直接放在run()中

class Solver:
    """多臂老虎机算法基本框架"""
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0 # 当前步的累积懊悔
        self.actions = [] # 维护一个列表，记录每一步的动作
        self.regrets = [] # 维护一个列表，记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError
    
    def run(self, num_steps):
        # 运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

"""在多臂老虎机问题中，一个经典的问题就是探索与利用的平衡问题。
   探索（exploration）是指尝试拉动更多可能的拉杆，这根拉杆不
   一定会获得最大的奖励，但这种方案能够摸清楚所有拉杆的获奖情况。
   例如，对于一个 10 臂老虎机，我们要把所有的拉杆都拉动一下才
   知道哪根拉杆可能获得最大的奖励。
   利用（exploitation）是指拉动已知期望奖励最大的那根拉杆，由于
   已知的信息仅仅来自有限次的交互观测，所以当前的最优拉杆不一定
   是全局最优的。例如，对于一个 10 臂老虎机，我们只拉动过其中 3 
   根拉杆，接下来就一直拉动这 3 根拉杆中期望奖励最大的那根拉杆，
   但很有可能期望奖励最大的拉杆在剩下的 7 根当中，即使我们对 10 
   根拉杆各自都尝试了 20 次，发现 5 号拉杆的经验期望奖励是最高
   的，但仍然存在着微小的概率—另一根 6 号拉杆的真实期望奖励是比
     5 号拉杆更高的。"""

# 算法1：ε-贪心算法

""" ε-贪婪算法在完全贪婪算法的基础上添加了噪声，每次以概率 1-ε
选择以往经验中期望奖励估值最大的那根拉杆（利用），以概率 ε 随机
选择一根拉杆（探索）"""

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法，继承solver类"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates) #选择期望奖励估值最大的拉杆

        r = self.bandit.step(k) # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    

# 自定义绘图函数

