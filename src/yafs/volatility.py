"""
This class is for setting the volatility of each node in the network.
"""

import logging
import random

class Volatility(object):
    # just choose some random starting point for these constants
    SOURCE = 1000
    PROXY = 1001
    SINK = 1002
    
    def __init__(self, app, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.app = app


    def set_unlinkdistr(self, distr):
        pass


    def get_unlinktime(self, message, vtype, topo=None, node=None):
        """ returrn a sample time between creation and unlink/ deletion, geven a tyep of node"""
        return 1.0


    def set_erasedistr(self, distr):
        pass


    def get_erasetime(self, message, vtypu, topo=None, node=None):
        """ return a sample  time between the unlinking of data and the erasure of the data. """
        return 1.0


class UniformVolatility(Volatility):
    """ Volatility drawn from a uniform distribution, set by each type of node. """

    def set_volatility(self, vtype, maxv, minv=0.0):
        self.vol[self.app][vtype] = (maxv, minv)

    def get_volatility(self, topo, node):
        lims = self.vol[self.app][topo.G[node].voltype]
        return random.uniform(lims[0], lims[1])


class FixedVolatility(Volatility):
    """ Just a single, set volatility for all nodes. For testing. """

    def get_unlinktime(self, message, vtype, topo=None, node=None):
        return self.ultime

    def get_erasetime(self, message, vtypu, topo=None, node=None):
        return self.etime

    def set_unlinkdistr(self, time):
        self.ultime = time

    def set_erasedistr(self, time):
        self.etime = time
