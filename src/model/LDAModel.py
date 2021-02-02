import numpy as np
from scipy.stats import dirichlet
import sys
sys.path.append('../util')
from utils import load_train_file

class LDAModel(object):
    def __init__(self, K):
        super(LDAModel, self).__init__()
        self.K = K
        self.data = None
        self.alpha = np.array([1.0 / self.K] * self.K)
        self.beta = None
        self.theta = None
        self.phi = None
        self.word2idx = {}
        self.z = None
        self.V = {}
        self.Vlen = None


    def fit(self, data):
        self.parse(data)
        self.initialization()

        """
        Gibbs sampling
        """
        for i in range(5):
            for i in range(self.M):
                for j in range(self.Nm[i]):
                    topic = self.sampling(i,j)
                    self.Z[i][j] = topic

        self._theta()
        self._phi()

    def _theta(self):
        for i in range(self.M):
            self.theta[i] = (self.mk[i] + self.alpha)*1.0/(np.sum(self.mk) + np.sum(self.alpha))


    def _phi(self):
        for j in range(self.K):
            self.phi[j] = (self.kt[j] + self.beta) * 1.0/(np.sum(self.kt) + np.sum(self.beta))

    def calculate_p(self,m,n,k):
        t = self.data[m][n]
        first_part_nom = self.beta[t] + self.kt[k][t]
        first_part_denom = np.sum(self.beta) + np.sum(self.k[k])

        second_part_nom = self.alpha[k] + self.mk[m][k]
        second_part_denom = np.sum(self.alpha) + np.sum(self.m[m])
        return first_part_nom * second_part_nom * 1.0/( first_part_denom * second_part_denom )

    def sampling(self,i,j):
        topic_k = self.Z[i][j]
        word = self.data[i][j]
        self.mk[i][topic_k] -= 1
        self.m[i] -= 1
        self.kt[topic_k][j] -= 1
        self.k[topic_k] -= 1

        self.p = [self.calculate_p(i,j,k) for k in range(self.K)]
        print(self.p)
        total_p = sum(self.p)
        self.p = [c*1.0/total_p for c in self.p]
        k_update = np.argmax(np.random.multinomial(1,self.p))

        self.mk[i][k_update] += 1
        self.m[i] += 1
        self.kt[k_update][word] += 1
        self.k[k_update] += 1
        return k_update



    def initialization(self):
        self.mk = np.zeros((self.M,self.K))
        self.m = np.zeros(self.M)
        self.kt = np.zeros((self.K,self.Vlen))
        self.k = np.zeros(self.K)

        for i in range(self.M):
            for j in range(self.Nm[i]):
                topic_k = np.argmax(np.random.multinomial(1,[1.0/self.K]*self.K))
                t = self.data[i][j]
                self.Z[i][j] = topic_k
                self.mk[i][topic_k] += 1
                self.m[i] += 1
                self.kt[topic_k][t] += 1
                self.k[topic_k]+=1

    def parse(self,data):
        self.data = data
        if self.data is None or not len(self.data):
            raise Exception("Error: data must not be empty")
        self.M = len(self.data)
        for doc in self.data:
            for term in doc:
                if term not in self.V:
                    self.V[term] = 0
                self.V[term] += 1
        self.Vlen = len(self.V)

        for word,idx in zip(list(self.V.keys()),list(range(self.Vlen))):
            self.word2idx[word] = idx

        self.Nm = np.zeros(self.M,dtype=np.int)
        for i in range(self.M):
            for j in range(len(self.data[i])):
                self.data[i][j] = self.word2idx[self.data[i][j]]

        for i in range(self.M):
            nm = len(self.data[i])
            self.Nm[i] = nm

        self.Z = [[0]*self.Nm[x] for x in range(self.M)]
        self.beta = np.array([1.0/self.Vlen] * self.Vlen)
        self.theta = np.zeros([self.M,self.K])
        self.phi = np.zeros([self.K,self.Vlen])


if __name__ == '__main__':
    data = load_train_file('../data/sohu_train.txt',line_num=1000)
    model = LDAModel(3)
    model.fit(data=data)