import proto

class OptimizerSWIG(object):
    '''C++ LTSS optimizer
    '''

    def __init__(self, g, h):
        self.N = len(g)
        self.g_c = proto.FArray()
        self.h_c = proto.FArray()
        self.g_c = g
        self.h_c = h

    def __call__(self):
        return proto.optimize_one__LTSS(self.N, self.g_c, self.h_c)
