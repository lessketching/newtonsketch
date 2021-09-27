import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time

from sketches import gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores

from sklearn.kernel_approximation import RBFSampler



SKETCH_FN = {'gaussian': gaussian, 'less': less, 'less_sparse': sparse_rademacher, 
             'srht': srht, 'rrs': rrs, 'rrs_lev_scores': rrs_lev_scores}


torch.set_default_dtype(torch.float64)

class LogisticRegression:
    
    def __init__(self, A, b, lambd):
        self.A = A
        self.b = b
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1,1))
        self.n, self.d = A.shape
        self.c = self.b.shape[1]
        self.lambd = lambd
        self.device = A.device
        
    def loss(self, x):
        return 1./2 * ((self.H_opt @ (x - self.x_opt))**2).sum()
    
    def logistic_loss(self, x):
        return (torch.log(1 + torch.exp(-self.b * (self.A @ x)))).mean() + self.lambd/2 * (x**2).sum()

    def grad(self, x, stochastic=False):
        if not stochastic:
            return -1./self.n*self.A.T @ ( self.b * 1./(1+torch.exp(self.b * (self.A @ x))))+ self.lambd * x
        else:
            index = np.random.choice(self.n)
            a_i = self.A[index,::].reshape((-1,1))
            b_i = self.b[index].squeeze()
            return -a_i.reshape((-1,1)) * (b_i /(1+torch.exp(b_i * ( (a_i*x).sum()))))+ self.lambd * x
        
    def hessian(self, x):
        Ax = self.A @ x
        v = torch.exp(self.b * Ax)
        D = v / (1+v)**2
        return 1./self.n * self.A.T @ (D * self.A) + self.lambd * torch.eye(self.d).to(self.device)

    def sqrt_hess(self, x):
        v_ = torch.exp(self.b * (self.A @ x))
        return 1./np.sqrt(self.n)*torch.sqrt(v_) / (1+v_)

    def line_search(self, x, v, g, alpha=0.3, beta=0.8):
        delta = (v*g).sum()
        loss_x = self.logistic_loss(x)
        s = 1
        xs = x + s*v
        while self.logistic_loss(xs) > loss_x + alpha*s*delta:
            s = beta*s 
            xs = x + s*v
        return s
    
    
    def solve_exactly(self, n_iter=100, eps=1e-8):
        losses = []
        x = 1./np.sqrt(self.d)*torch.randn(self.d, self.c).to(self.device)
        
        for _ in range(n_iter):
            losses.append(self.logistic_loss(x).cpu().numpy().item())
            g = self.grad(x)
            H = self.hessian(x)
            v = -torch.pinverse(H) @ g
            delta = - (g * v).sum()
            if delta < eps:
                break
            s = self.line_search(x, v, g)
            x = x + s*v
        
        self.x_opt = x
        self.H_opt = H
        
        _, sigma, _ = torch.svd(self.H_opt)
        
        de_ = torch.trace((H - self.lambd * torch.eye(self.d).to(self.device)) @ torch.pinverse(H))
        self.de = de_.cpu().numpy().item()
        
        return x, losses
        
        
    def newton(self, n_iter=10):
        losses = []
        times = []
        
        x = 1./np.sqrt(self.d)*torch.randn(self.d, self.c).to(self.device)
        
        for _ in range(n_iter):
            losses.append(self.loss(x).cpu().numpy().item())
            
            start = time()
            g = self.grad(x)
            H = self.hessian(x)
            v = -torch.pinverse(H) @ g
            delta = - (g * v).sum()
            s = self.line_search(x, v, g)
            x = x + s*v
            times.append(time()-start)
        
        losses = np.array(losses)
        losses /= losses[0]
        
        return x, losses, np.cumsum([0]+times)[:-1]
    
    
    def ihs_(self, x, sketch_size, sketch, nnz):
        
        start = time()
        
        hsqrt = self.sqrt_hess(x).reshape((-1,1))
        sa = SKETCH_FN[sketch](hsqrt * self.A, sketch_size, nnz=nnz)
        g = self.grad(x)
        
        if sketch_size >= self.d:
            hs = sa.T @ sa + self.lambd * torch.eye(self.d).to(self.device)
            u = torch.cholesky(hs)
            v = -torch.cholesky_solve(g, u)
        elif sketch_size < self.d:
            ws = sa @ sa.T + self.lambd * torch.eye(sketch_size).to(self.device)
            u = torch.cholesky(ws)
            sol_ = torch.cholesky_solve(sa @ g, u)
            v = -1./self.lambd * (g - sa.T @ sol_)
            
        #v = -torch.pinverse(hs) @ g
        s = self.line_search(x, v, g)
        x = x + s*v
        
        return x, time()-start
        
        
    
    def ihs(self, sketch_size, sketch='gaussian', nnz=1., n_iter=10):
        
        losses = []
        times = []
        
        #time_sketch = profile_times(A, sketch, sketch_size, nnz)
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        for _ in range(n_iter):
            losses.append(self.loss(x).cpu().numpy().item())
            x, time_ = self.ihs_(x, sketch_size, sketch, nnz)
            times.append(time_)
        
        losses = np.array(losses)
        losses /= losses[0]
        
        return x, losses, np.cumsum([0] + times)[:-1]
    
    
    
    def gd(self, n_iter=1000):      
        losses = []
        times = []
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        for _ in range(n_iter):
            losses.append(self.loss(x).cpu().numpy().item())
            
            start = time()
            
            g = self.grad(x)
            s = self.line_search(x, -g, g)
            x = x - s*g
            
            times.append(time()-start)
        
        losses = np.array(losses)
        losses /= losses[0]
            
        return x, losses, np.cumsum([0] + times)[:-1]
    
    
    
    def sgd(self, n_iter=100, s=0.01):
        losses = []
        times = []
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        
        for _ in range(n_iter):
            
            losses.append(self.loss(x).cpu().numpy().item())
            
            start = time()
            
            g_sto = self.grad(x, stochastic=True)
            x = x - s*g_sto
            
            times.append(time()-start)
            
        losses = np.array(losses)
        losses /= losses[0]
            
        return x, losses, np.cumsum([0] + times)[:-1]
        
        
        
    
    def bfgs(self, n_iter=10):
        
        losses = []
        times = []
        
        x = 1./np.sqrt(self.d) * torch.randn(self.d,1).to(self.device)
        Binv = torch.eye(self.d).to(self.device)
        
        g = self.grad(x)
        
        for _ in range(n_iter):
            
            losses.append(self.loss(x).cpu().numpy().item())
            
            start = time()
            
            v = - Binv @ g
            mu = self.line_search(x, v, g)
            s = mu * v
            x = x + s 
            g_ = self.grad(x)
            y = g_ - g
            g = g_.clone()
            sy_inner = (s * y).sum()
            sy_outer = s.reshape((-1,1)) @ y.reshape((1,-1))
            ss_outer = s.reshape((-1,1)) @ s.reshape((1,-1))
            B_1 = (sy_inner + (y* (Binv@y)).sum())/sy_inner**2 * ss_outer
            b_2 = Binv @ sy_outer.T
            B_2 = (b_2 + b_2.T) / sy_inner
            Binv = Binv + B_1 - B_2
            
            times.append(time()-start)
            
        losses = np.array(losses)
        losses /= losses[0]
        
        return x, losses, np.cumsum([0] + times)[:-1]
    


'''
Example code

n = 16000
d = 6000 
lambd = 1e-5

A = np.random.randn(n,d)
u, sigma, vh = np.linalg.svd(A, full_matrices=False)
sigma = np.array([0.98**jj for jj in range(d)])
A = u @ (np.diag(sigma) @ vh)

m = 300
nnz = 0.02

xpl = 1./np.sqrt(d)*np.random.randn(d,1)
b = np.sign(A@ xpl)

A = torch.tensor(A)
b = torch.tensor(b)

lreg = LogisticRegression(A, b, lambd)

x, losses = lreg.solve_exactly(n_iter=20, eps=1e-15)

m = 50

n_iter_gd = 500
n_iter_sgd = 500
n_iter_newton = 5
n_iter_ihs = 30
n_iter_bfgs = 100

nnz = 0.005

losses_ihs = {}
times_ihs = {}

sketches = ['less_sparse', 'gaussian', 'rrs', 'srht']

_, losses_newton, times_newton = lreg.newton(n_iter=n_iter_newton)
_, losses_gd, times_gd = lreg.gd(n_iter=n_iter_gd)
_, losses_sgd, times_sgd = lreg.sgd(n_iter=n_iter_sgd, s=0.001)
_, losses_bfgs, times_bfgs = lreg.bfgs(n_iter=n_iter_bfgs)

for sketch in sketches:
    print('ihs: ', sketch)
    _, losses_, times_ = lreg.ihs(sketch_size=m, sketch=sketch, nnz=nnz, n_iter=n_iter_ihs)
    losses_ihs[sketch] = losses_
    times_ihs[sketch] = times_

'''














    
    
    
    
