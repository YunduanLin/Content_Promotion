import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import CGPO

class platform:
    def __init__(self, param):
        self.t = 0
        self.gamma = param['gamma']
        self.m, self.K = param['m'], param['K']
        self.m_init = self.m
        self.cnt_new = param['cnt_new']
        self.cnt = 0
        self.p, self.q = np.array([]), np.array([])
        self.A = np.array([],dtype='int32')
        self.t_v = np.array([],dtype='int32')
        self.x = np.array([])
        self.a_mix, self.a_diff = np.array([],dtype='int32'), np.array([],dtype='int32')
        df_cat = pd.read_csv(param['file_pq'])
        self.p_all, self.q_all = df_cat['p'].values, df_cat['q'].values

    def add_new_v(self, p, q, A):
        self.p, self.q = np.append(self.p, p), np.append(self.q, q)
        self.A = np.append(self.A, A)
        self.t_v = np.append(self.t_v, 0)
        self.x = np.append(self.x, 0)
        self.a_mix, self.a_diff = np.append(self.a_mix, 0), np.append(self.a_diff, 0)

    def initialize(self, cnt):
        for i in range(cnt):
            self.add_new_v(self.p_all[i], self.q_all[i], 0)
        self.cnt = cnt

    def sim_PO(self, param):
        L, C, T = param['L'], param['C'], param['T']
        alg = param['alg']

        m_new = param['m_new']

        f_A, f_x = open(f'./{alg}_L{L}_c{C//L}/A.txt', 'a'), open(f'./{alg}_L{L}_c{C//L}/x.txt', 'a')
        f_a_mix, f_a_diff = open(f'./{alg}_L{L}_c{C//L}/a_mix.txt', 'a'), open(f'./{alg}_L{L}_c{C//L}/a_diff.txt', 'a')

        for t in range(T):
            self.m += m_new
            self.t += 1
            for i in range(self.cnt, self.cnt + self.cnt_new):
                self.add_new_v(self.p_all[i], self.q_all[i], 0)
            self.cnt += self.cnt_new

            if t % L == 0:
                param_PO = {'p': self.p, 'q': self.q, 'K': self.K, 'L': L, 'C': C, 'gamma':self.gamma}
                opt = CGPO.CGPO(param_PO)
                if alg=='ATT':
                    ind = np.argsort(self.p*(self.m-self.A))[-self.K:]
                elif alg=='NEW':
                    ind = np.arange(self.cnt-self.K,self.cnt,1).astype(int)
                elif alg=='POT':
                    ind = np.argsort(self.m-self.A)[-self.K:]

                param_PO = {'A0': self.A / self.m, 't0': self.t_v, 'ind':ind}
                opt.optimize_PO(param_PO)

                x = opt.x

            x_cur = x[:, 0]

            self.sim_one_step(ind, x_cur)
            x = x[:, 1:]

            print(f'Time {self.t}: total adoption {np.sum(self.A)}')

            f_A.write(','.join(self.A.astype(str)))
            f_A.write('\n')
            f_x.write(','.join(self.x.astype(str)))
            f_x.write('\n')
            f_a_mix.write(','.join(self.a_mix.astype(str)))
            f_a_mix.write('\n')
            f_a_diff.write(','.join(self.a_diff.astype(str)))
            f_a_diff.write('\n')

            self.t_v = self.t_v + 1

        f_A.close()
        f_x.close()
        f_a_mix.close()
        f_a_diff.close()

    def sim_CGPO(self, param):
        L, C, T = param['L'], param['C'], param['T']

        m_new = param['m_new']

        f_A, f_x = open(f'./CGPO_L{L}_c{C//L}/A.txt', 'a'), open(f'./CGPO_L{L}_c{C//L}/x.txt', 'a')
        f_a_mix, f_a_diff = open(f'./CGPO_L{L}_c{C//L}/a_mix.txt', 'a'), open(f'./CGPO_L{L}_c{C//L}/a_diff.txt', 'a')

        for t in range(T):
            self.m += m_new
            self.t += 1
            for i in range(self.cnt, self.cnt + self.cnt_new):
                self.add_new_v(self.p_all[i], self.q_all[i], 0)
            self.cnt += self.cnt_new

            if t % L == 0:
                param_CGPO = {'p': self.p, 'q': self.q, 'K': self.K, 'L': L, 'C': C, 'gamma':self.gamma}
                opt = CGPO.CGPO(param_CGPO)
                param_CGPO = {'A0': self.A / self.m, 't0': self.t_v}
                opt.optimize(param_CGPO)

                ind, x = opt.ind, opt.x

            x_cur = x[:, 0]

            self.sim_one_step(ind, x_cur)
            x = x[:, 1:]

            print(f'Time {self.t}: total adoption {np.sum(self.A)}')

            f_A.write(','.join(self.A.astype(str)))
            f_A.write('\n')
            f_x.write(','.join(self.x.astype(str)))
            f_x.write('\n')
            f_a_mix.write(','.join(self.a_mix.astype(str)))
            f_a_mix.write('\n')
            f_a_diff.write(','.join(self.a_diff.astype(str)))
            f_a_diff.write('\n')

            self.t_v = self.t_v + 1

        f_A.close()
        f_x.close()
        f_a_mix.close()
        f_a_diff.close()

    def sim_one_step(self, ind, x):
        ind_bool = np.zeros(self.cnt,dtype=bool)
        ind_bool[ind] = True
        ind_no_prom = np.arange(0,self.cnt,1)[~ind_bool]

        ind = np.append(ind, ind_no_prom)
        x = np.append(x, [0]*len(ind_no_prom))

        # promote
        A, p, q = self.A[ind], self.p[ind], self.q[ind]
        # mixing effect:
        n_mix = np.array(list(map(int, self.m * x)))
        n_mix = np.maximum(np.minimum(n_mix, self.m - A, dtype='int32'), 0, dtype='int32')
        prob_mix = p + q * self.gamma ** self.t_v * A / self.m
        a_mix = np.random.binomial(n_mix, prob_mix)

        # diffusion effect:
        n_diff = self.m - A - n_mix
        n_diff = np.maximum(n_diff, 0, dtype='int32')
        prob_diff = q * self.gamma ** self.t_v * A / self.m
        a_diff = np.random.binomial(n_diff, prob_diff)

        A = A + a_mix + a_diff
        self.A[ind] = A

        self.x[ind] = x
        self.a_mix[ind], self.a_diff[ind] = a_mix, a_diff

    def reset(self):
        self.m = self.m_init
        self.t = 0
        self.p, self.q = np.array([]), np.array([])
        self.A = np.array([], dtype='int32')
        self.t_v = np.array([], dtype='int32')
        self.x = np.array([])
        self.a_mix, self.a_diff = np.array([], dtype='int32'), np.array([], dtype='int32')
