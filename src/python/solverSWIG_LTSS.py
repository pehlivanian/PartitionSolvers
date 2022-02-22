import proto

class Distribution:
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

class OptimizerSWIG(object):
    '''C++ LTSS optimizer
    '''

    def __init__(self, g, h, objective_fn=Distribution.GAUSSIAN):
        self.N = len(g)
        self.g_c = proto.FArray()
        self.h_c = proto.FArray()
        self.g_c = g
        self.h_c = h
        self.objective_fn = objective_fn

    def __call__(self):
        return proto.optimize_one__LTSS(self.N, self.g_c, self.h_c, self.objective_fn)
