import math
import numpy as np
import pandas as pd
from scipy.special import logsumexp, entr
from scipy.stats.contingency import margins
np.seterr(divide='ignore', invalid="ignore")

PRECISION = 1e-16

# PROBABILITY FUNCTIONS
def marginal(pXY, axis=1):
    """Return the marginal distribution of pXY.
    
    Return pY if axis = 0 or pX (default) if axis = 1.
    """
    return pXY.sum(axis)

def conditional(pXY):
    """Return the conditional probability of pX|Y."""
    pX = pXY.sum(axis=1, keepdims=True)
    return np.where(pX > PRECISION, pXY / pX, 1 / pXY.shape[1])

def joint(pY_X, pX):
    """Return the joint probability of pXY given pY|X and pX."""
    return pY_X * pX[:, None]

def marginalize(pY_X, pX):
    """Return marginal distribution of Y."""
    return pY_X.T @ pX

def bayes(pY_X, pX):
    """:return pX_Y """
    pXY = joint(pY_X, pX)
    pY = marginalize(pY_X, pX)
    return np.where(pY > PRECISION, pXY.T / pY, 1 / pXY.shape[0])

def softmax(dxy, beta=1, axis=None):
    """:return
        axis = None: pXY propto exp(-beta * dxy)
        axis = 1: pY_X propto exp(-beta * dxy)
        axis = 0: pX_Y propto exp(-beta * dxy)
    """
    log_z = logsumexp(-beta * dxy, axis, keepdims=True)
    return np.exp(-beta * dxy - log_z)


# INFORMATIONAL MEASURES
def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """ mutual information, I(X;Y) """
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def entropy_complexity(pW_M):
    """Calculate the complexity of the encoder system with respect to
    entropy.
    
    The sum entropy over every word."""
    normalized_pW_M = pW_M / pW_M.sum(axis=0, keepdims=0)
    return np.nansum(entr(normalized_pW_M))

    
def complexity_split(pW_M, pM):
    """
    :param pW_M: encoder (naming system)
    :return: I(M;W)
    """
    return MI(pW_M * pM)


def accuracy(pW_M, pM, pU_M):
    """
    :param pW_M: encoder (naming system)
    :return: I(W;U)
    """
    pMW = pW_M * pM
    pWU = pMW.T @ pU_M
    return MI(pWU)

def m_hat(qW_M, pM, pU_M):
    """
    :param qW_M: encoder (naming system)
    :return: optimal decoder (Bayesian listener) that corresponds to the encoder
    """
    # Each row is a meaning, each column is a word
    pMW = qW_M * pM
    pM_W = np.nan_to_num(pMW.T / pMW.sum(axis=0)[:, None], 0)
    return pM_W.dot(pU_M)


def DKL(p, q, axis=None):
    """ KL divergences, D[p||q] """
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(axis=axis)


def calculate_KL(curr_obj):
    """Calculate the KL divergence"""
    curr_pm, curr_pw, beta, curr_mhat = curr_obj
    div = np.log2(curr_pw)-beta*DKL(curr_pm, curr_mhat, axis=1)
    div = (div - logsumexp(div)).reshape(-1)
    return np.exp2(div)


class IBObj:
    def __init__(self, com_name, joint, pu_m):
        """
        Holds all information related to the efficiency calculation for a particular community.
        Args:
            com_name (str): the community that efficiency is being calculated for
        """
        self.com_name = com_name
        self.joint = joint
        np.testing.assert_almost_equal(np.sum(self.joint), 1)
        self.p_m, self.p_w = margins(self.joint)
        self.encoder = np.nan_to_num(self.joint/self.p_m)
        self.pu_m = pu_m
        self.decoder = m_hat(self.encoder, self.p_m, self.pu_m)
        self.complexity = complexity_split(self.encoder, self.p_m)
        self.entropy_complexity = entropy_complexity(self.encoder)
        self.accuracy = accuracy(self.encoder, self.p_m, self.pu_m)
        
        
    def get_accuracy(self):
        return accuracy(self.encoder, self.p_m, self.pu_m)
    
    
    def get_complexity(self):
        return complexity_split(self.encoder, self.p_m)
    
    
    def get_entropy_complexity(self):
        return entropy_complexity(self.encoder)


def pandas_entropy(column, base=None):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = math.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()