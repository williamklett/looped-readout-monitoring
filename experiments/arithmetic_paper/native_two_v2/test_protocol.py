import unittest
from itertools import product
from types import SimpleNamespace
import torch
from torch import nn
from protocol import stopping_weights,cells
from core import two_digit_loss,prepare_data

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.arange(10,dtype=torch.float32))
        self.base=SimpleNamespace(lm_head=nn.Identity());self.calls=[];self.depth=3
    def forward(self,ids,attention_mask=None,last_only=False):
        self.calls.append(ids.tolist())
        z=self.theta.view(1,1,-1).expand(ids.shape[0],ids.shape[1],-1)
        return z,[z]*self.depth

class Tests(unittest.TestCase):
    def test_no_second_answer_input_or_eos_loss(self):
        model=Toy();rows=[dict(ids=[1,2,3],target_ids=[4,5])]
        loss,ce,pen=two_digit_loss(model,rows,0,[4,5],2.)
        self.assertEqual(model.calls,[[[1,2,3,4]]])
        lp=model.theta.log_softmax(-1)
        torch.testing.assert_close(ce,-(lp[4]+lp[5])/2)
        torch.testing.assert_close(pen,model.theta.softmax(-1)[[4,5]].sum())
        loss.backward();self.assertGreater(float(model.theta.grad.norm()),0)
    def test_first_depth_has_no_penalty(self):
        m=Toy();m.depth=1
        _,_,p=two_digit_loss(m,[dict(ids=[1,2],target_ids=[4,5])],0,[4,5],100.)
        self.assertEqual(float(p),0.)
    def test_shared_depth_same_additive_expectation(self):
        for hazard in [.25,.5,.75]:
            q,w=stopping_weights(hazard)
            a=[.2+k*.13 for k in range(7)];b=[1.1-k*.09 for k in range(7)]
            independent=sum(q[i]*q[j]*(a[i]+b[j])/2 for i,j in product(range(7),repeat=2))
            shared=sum(q[i]*(a[i]+b[i])/2 for i in range(7))
            self.assertAlmostEqual(independent,shared)
            self.assertAlmostEqual(sum(q),1.)
            c=[.1+i*.1 for i in range(7)]
            self.assertAlmostEqual(sum(q[n-1]*sum(c[:n-1])/(n-1) for n in range(2,8)),sum(x*y for x,y in zip(w,c)))
    def test_grid_and_derivative(self):
        self.assertEqual(len(cells()),45)
        for cell in cells():
            if not cell['penalty_weight']:continue
            q,w=stopping_weights(cell['stop_probability']);lam=cell['penalty_weight']
            for k,c in enumerate(cell['predicted_mass']):
                if c<1:self.assertAlmostEqual(-q[k]/c+lam*w[k],0.)
            self.assertAlmostEqual(cell['predicted_mass'][3],.1/cell['boundary_factor'])

if __name__=='__main__':unittest.main()
