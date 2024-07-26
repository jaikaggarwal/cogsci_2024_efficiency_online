import numpy as np
import unittest
from ..info_theory_utils import *

class InfoTheoryTestCase(unittest.TestCase):
    def setUp(self):
        self.joint = np.array([[4/9, 1/9], [2/9, 2/9]])
        self.conditionalY = np.array([[2/3, 1/3], [1/3, 2/3]])
        self.pX = np.array([2/3, 1/3])
        
        
    def testMarginalX(self):
        marginalX = marginal(self.joint)
        solxn = np.array([1.0, 1.0])
        self.assertEqual(marginalX, solxn)
        
        
    def testMarginalY(self):
        marginalY = marginal(self.matrix, axis=0)
        solxn = np.array([1.0, 1.0])
        self.assertEqual(marginalY, solxn)


    def testConditionalX_Y(self):
        conditionalX = conditional(self.joint)
        solxn = np.array([[0.8, 0.2], [0.5, 0.5]])
        self.assertEqual(conditionalX, solxn)


    def testJoint(self):
        pXY = joint(self.conditionalY, self.pX) 
        self.assertEqual(pXY, self.joint)


    def testMarginalize(self):
        marginalY = marginalize(self.conditionalY, self.pX)
        solxn = np.array([5/9, 4/9])
        self.assertEqual(marginalY, solxn)


    def testBayes(self):
        pX_Y = bayes(self.conditionalY, self.pX)
        solxn = np.array([[0.8, 0.25], [0.4, 0.5]])
        self.assertEqual(pX_Y, solxn)
        
        
    def testxlogx(self):
        x = np.array([1, 2, 3, 1e-17])
        test_xlogx = xlogx(x)
        solxn = np.array([0, 2, 3 * np.log2(3), 0])
        self.assertEqual(test_xlogx, solxn)
        
    
    def testHighEntropy(self):
        pXY = np.array([[0.5, 0.5], [0.5, 0.5]])
        test_entropy = H(pXY)
        solxn = 2.0
        self.assertEqual(test_entropy, solxn)
        
    
    def testLowEntropy(self):
        pXY = np.array([[1.0, 0.], [0., 1.0]])
        test_entropy = H(pXY)
        solxn = 0.0
        self.assertEqual(test_entropy, solxn)
        
    