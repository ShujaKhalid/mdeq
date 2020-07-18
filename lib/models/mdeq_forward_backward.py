# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../")
from modules.deq2d import *

__author__ = "shaojieb"


class MDEQWrapper(DEQModule2d):
    def __init__(self, func, func_copy):
        super(MDEQWrapper, self).__init__(func, func_copy)
        self.df_dx = 0
    
    def forward(self, z1, u, **kwargs):
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)

        if u is None:
            raise ValueError("Input injection is required.")

        new_z1 = list(DEQFunc2d.apply(self.func, z1, u, threshold, train_step, writer))
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in new_z1]

        if self.training:
            new_z1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1, u, threshold, train_step))
            new_z1 = self.Backward.apply(self.func_copy, new_z1, u, threshold, train_step, writer)

            '''
            Calculate dF/dx
            '''
            z1_temp = new_z1.clone().detach().requires_grad_()
            with torch.enable_grad():
                f_x = DEQFunc2d.f_x(self.func, z1_temp, u, cutoffs, train_step)

            def f(x):
                f_x.backward(x, retain_graph=True)
                df_dx = z1_temp.grad.clone()
                z1_temp.grad.zero_()
                return df_dx

            # Here is your grad_f_x
            self.df_dx = f(z1_temp)
            #print()
            #print('===> Backward <===')
            #print('df_dx_norm: {}'.format(torch.norm(self.df_dx)))

            #torch.cuda.empty_cache()

            # Delete graph
            #f_x.backward(torch.zeros_like(z1_temp), retain_graph=False)

            new_z1 = DEQFunc2d.vec2list(new_z1, cutoffs)
        return new_z1

