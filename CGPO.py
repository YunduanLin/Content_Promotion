import numpy as np
import numpy.linalg as la
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


class CGPO:
    def __init__(self, param):
        self.p, self.q = param['p'], param['q']
        self.K, self.L, self.C = param['K'], param['L'], param['C']
        self.gamma = param['gamma']

    def optimize_PO(self,param):
        A0 = param['A0']
        t0 = param['t0']

        ind = param['ind']

        mod = gp.Model('PO')
        mod.Params.LogToConsole = 0

        x, A = np.zeros(0, dtype=object), np.zeros(0, dtype=object)

        for k in range(len(ind)):
            x = np.append(x, mod.addVars(self.L+1, vtype=GRB.CONTINUOUS, lb=0, ub=1,name='x'))
            A = np.append(A, mod.addVars(self.L+1, vtype=GRB.CONTINUOUS, lb=0, ub=1,name='A'))
            x[-1][0].lb, x[-1][0].ub = 0, 0
            A[-1][0].lb, A[-1][0].ub = A0[ind[k]], A0[ind[k]]

            constr_ub = mod.addConstrs(x[-1][t] <= 1 - A[-1][t - 1] for t in range(1, self.L + 1))

            constr_diff = mod.addConstrs(
                A[-1][t] <= A[-1][t - 1] + self.p[ind[k]] * x[-1][t]
                + self.q[ind[k]] * self.gamma ** (t0[ind[k]]+t-1) * A[-1][t - 1] * (1 - A[-1][t - 1])
                 for t in range(1, self.L + 1))

        constr_cap = mod.addConstr(np.sum([[x[i][t] for t in range(1, self.L + 1)] for i in range(len(ind))]) <= self.C)

        mod.setObjective(np.sum([A[i][self.L] for i in range(len(ind))]), GRB.MAXIMIZE)
        mod.update()
        mod.optimize()

        self.x = np.zeros((len(ind), self.L))
        self.A = np.zeros(len(ind))
        for i in range(len(ind)):
            for t in range(1,self.L+1):
                self.x[i, t-1] = x[i][t].x
            self.A[i] = A[i][self.L].x


    def optimize(self, param):
        A0 = param['A0']
        t0 = param['t0']

        Ax0 = A0.copy().astype('float64')
        for t in range(1,self.L+1):
            a = self.q * self.gamma ** (t0+t-1) * Ax0 * (1-Ax0)
            Ax0 += a

        ind = np.array([],dtype='int64')

        mod = gp.Model('CGPO')
        mod.Params.LogToConsole = 0
        x, A = np.zeros(0, dtype=object), np.zeros(0, dtype=object)

        for k in range(self.K):
            if k>=len(A0):
                break
            x = np.append(x, mod.addVars(self.L+1, vtype=GRB.CONTINUOUS, lb=0, ub=1,name='x'))
            A = np.append(A, mod.addVars(self.L+1, vtype=GRB.CONTINUOUS, lb=0, ub=1,name='A'))
            x[-1][0].lb, x[-1][0].ub = 0, 0

            constr_cap = mod.addConstr(np.sum([[x[i][t] for t in range(1, self.L + 1)] for i in range(k+1)]) <= self.C)
            constr_ub = mod.addConstrs(x[-1][t] <= 1 - A[-1][t - 1] for t in range(1, self.L + 1))

            ind_cur = 0
            res = np.array([])

            while ind_cur<len(self.p):

                if ind_cur in ind:
                    res = np.append(res,0)
                    ind_cur += 1
                    continue

                A[-1][0].lb, A[-1][0].ub = A0[ind_cur], A0[ind_cur]

                mod.setObjective(np.sum([A[i][self.L] for i in range(k+1)]), GRB.MAXIMIZE)

                constr_diff = mod.addConstrs(
                    A[-1][t] <= A[-1][t - 1] + self.p[ind_cur] * x[-1][t]
                    + self.q[ind_cur] * self.gamma ** (t0[ind_cur]+t-1) * A[-1][t - 1] * (1 - A[-1][t - 1])
                     for t in range(1, self.L + 1))

                mod.update()
                mod.optimize()

                res = np.append(res, mod.objVal + np.sum(Ax0)-Ax0[ind_cur]-np.sum(Ax0[ind]))

                mod.remove(constr_diff)
                ind_cur += 1


            ind_max = np.argmax(res)
            ind = np.append(ind, ind_max)

            A[-1][0].lb, A[-1][0].ub = A0[ind_max], A0[ind_max]
            mod.addConstrs(
                A[-1][t] <= A[-1][t - 1] + self.p[ind_max] * x[-1][t]
                + self.q[ind_max] * self.gamma ** (t0[ind_max]+t-1) * A[-1][t - 1] * (1 - A[-1][t - 1]) for t
                 in range(1, self.L + 1))

            mod.remove(constr_cap)

        cnt = min(self.K, len(A0))
        mod.setObjective(np.sum([A[i][self.L] for i in range(cnt)]), GRB.MAXIMIZE)
        constr_cap = mod.addConstr(np.sum([[x[i][t] for t in range(1, self.L + 1)] for i in range(cnt)]) <= self.C)

        mod.update()
        mod.optimize()

        self.ind = ind
        self.mod = mod

        self.x = np.zeros((cnt, self.L))
        self.A = np.zeros(cnt)
        for i in range(cnt):
            for t in range(1,self.L+1):
                self.x[i, t-1] = x[i][t].x
            self.A[i] = A[i][self.L].x
