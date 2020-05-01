import numpy as np

def emptyFunc(x=None,e=None,xx=None,ee=None,xxx=None,eee=None):
    if x is not None: return x
    if xx is not None: return xx
    if xxx is not None: return xxx

def ReLU(x=None,xx=None,xxx=None):
    if x is not None:
      if x > 0.0: return x
      else: return 0.0

    elif xx is not None:
      rr = []
      for r in xx:
        rr.append(ReLU(x=r))
      return rr

    elif xxx is not None:
      rrr = []
      for rr in xxx:
        rrr.append(ReLU(xx=rr))
      return np.array(rrr)

def derivReLU(x=None,xx=None,xxx=None):
    if x is not None:
      if x > 0.0: return 1.0
      else: return 0.0

    elif xx is not None:
      rr = []
      for r in xx:
        rr.append(derivReLU(x=r))
      return rr

    elif xxx is not None:
      rrr = []
      for rr in xxx:
        rrr.append(derivReLU(xx=rr))
      return np.array(rrr)

def sigmoid(x=None,xx=None,xxx=None):
    if x is not None:
        return 1/(1 + np.exp(x))

    elif xx is not None:
        rr = []
        for r in xx:
            rr.append(sigmoid(x=r))
        return rr

    elif xxx is not None:
        rrr = []
        for rr in xxx:
            rrr.append(sigmoid(xx=rr))
        return np.array(rrr)

def derivSigmoid(x=None,xx=None,xxx=None):
    if x is not None:
        return x*(1-x)

    elif xx is not None:
        rr = []
        for r in xx:
            rr.append(derivSigmoid(x=r))
        return rr

    elif xxx is not None:
        rrr = []
        for rr in xxx:
            rrr.append(derivSigmoid(xx=rr))
        return np.array(rrr)

def tanh(x=None,xx=None,xxx=None):
    if x is not None:
        return np.tanh(x)

    elif xx is not None:
      return np.tanh(xx)

    elif xxx is not None:
      return np.tanh(xxx)

def derivTanh(x=None,xx=None,xxx=None):
    if x is not None:
        return 1 - (x ** 2)

    elif xx is not None:
        rr = []
        for r in xx:
            rr.append(derivTanh(x=r))
        return rr

    elif xxx is not None:
        rrr = []
        for rr in xxx:
            rrr.append(derivTanh(xx=rr))
        return np.array(rrr)

def softmax(x=None,xx=None,xxx=None):
    if x is not None:
        return np.exp(x)
    elif xx is not None:
        return np.exp(xx)/sum(np.exp(xx))
    elif xxx is not None:
        ret = []
        for rr in xxx:
            ret.append(softmax(xx=rr))
        return ret

def derivSoftmax(x=None,e=None,xx=None,ee=None,xxx=None,eee=None):
    if x is not None and e is not None:
        return x - e
    elif xx is not None and ee is not None:
        return (xx - ee)/len(ee)
    elif xxx is not None and eee is not None:
        ret = []
        for i in range(len(xxx)):
            ret.append(derivSoftmax(xx=xxx[i], ee=eee[i]))
        return ret