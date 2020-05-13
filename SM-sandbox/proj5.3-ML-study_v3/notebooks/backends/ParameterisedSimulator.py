#   Implementation of ParameterisedSimulator base class
#   Author:  Stephen Menary  (stmenary@cern.ch)

import numpy as np
from   scipy import stats

from .Simulator import Simulator
from .stats     import randomly_sample_function, randomly_sample_grid



#  Class:  Generate random datapoints according to an analytic generative model
#          - model is hard-coded for now
#
class ParameterisedSimulator (Simulator) :
    def __init__ (self, name="", function=None) :
        self.clear()
        if len(name) > 0 :
            self.name = name
        self.set_function(function)
    def clear (self) :
        super(ParameterisedSimulator, self).clear()
        self.name     = ""
        self.function = None
    def set_function (self, generator) :
        self.generator = generator



#  Create some specific models to play with
#

#  Brief: return {datapoints}, with datapoints in the format {tot-xsec, dphi}
#
def generator_model1 (self, n_points, *argv, **kwargs) :
    params    = {p.name:p.value for p in self.params}
    c         = params["cX"]
    SM_xsec   = 2.
    int_xsec  = -1.*c
    c_xsec    = 1.*c*c
    tot_xsec  = SM_xsec + int_xsec + c_xsec
    dphi      = np.linspace(-1.*np.pi, 1.*np.pi, 6900)
    prob_dphi = SM_xsec + c_xsec*np.sin(dphi)
    return np.array([weights,
                     randomly_sample_function (dphi, prob_dphi, n_points)]).transpose()

Simulator_Model1 = ParameterisedSimulator(name="Model1")
Simulator_Model1.add_param ("cX")
Simulator_Model1.set_function(generator_model1)


#  Brief: return {tot-xsec, datapoints}, with datapoints in the format {A, B, C}
#
def generator_model2 (self, n_points, *argv, **kwargs) :
    params    = {p.name:p.value for p in self.params}
    mu        = params["mu"]
    SM_xsec   = 2.
    BSM_xsec  = 0.3*mu
    tot_xsec  = SM_xsec + BSM_xsec

    axis_A    = np.linspace(50, 250, 101)
    axis_B    = np.linspace(50, 500, 101)
    axis_C    = np.linspace(-1.*np.pi, np.pi, 101)

    if hasattr(self, "PDF") == False :
        
        print("A")

        PDF_A_SM  = stats.cauchy.pdf(axis_A, 20, 70)
        PDF_A_SM  = PDF_A_SM / np.sum(PDF_A_SM)
        PDF_A_BSM = pow((axis_A / 200.),3) * stats.cauchy.pdf(axis_A, 20, 70)
        PDF_A_BSM = PDF_A_BSM / np.sum(PDF_A_BSM)

        print("B")

        PDF_B_SM  = 1 / ((axis_B+300)*(axis_B+300)*(axis_B+300)*(axis_B+300))
        PDF_B_SM  = PDF_B_SM / np.sum(PDF_B_SM)
        PDF_B_BSM = stats.norm.pdf(axis_B, 250, 25)
        PDF_B_BSM = PDF_B_BSM / np.sum(PDF_B_BSM)

        print("C")

        PDF_C_SM  = np.ones(axis_C.shape)
        PDF_C_SM  = PDF_C_SM / np.sum(PDF_C_SM)
        PDF_C_BSM = 1.5 - np.cos(axis_C)
        PDF_C_BSM = PDF_C_BSM / np.sum(PDF_C_BSM)

        print("PDF")

        PDF_SM  = np.zeros(shape=(len(axis_A), len(axis_B), len(axis_C)))
        PDF_BSM = np.zeros(shape=(len(axis_A), len(axis_B), len(axis_C)))
        for i, (A, pA_SM, pA_BSM) in enumerate(zip(axis_A, PDF_A_SM, PDF_A_BSM)) :
            print(i)
            for j, (B, pB_SM, pB_BSM) in enumerate(zip(axis_B, PDF_B_SM, PDF_B_BSM)) :
                for k, (C, pC_SM, pC_BSM) in enumerate(zip(axis_C, PDF_C_SM, PDF_C_BSM)) :
                    corr_AB = stats.norm.pdf(((B-250.)-(A-70.)), 20.)
                    PDF_SM  [i, j, k] = pA_SM  * pB_SM  * pC_SM  * corr_AB
                    PDF_BSM [i, j, k] = pA_BSM * pB_BSM * pC_BSM * corr_AB

        print("normalise")

        PDF_SM  = PDF_SM  / np.sum(PDF_SM )
        PDF_BSM = PDF_BSM / np.sum(PDF_BSM)

        print("combine")

        self.PDF = np.zeros(shape=(len(axis_A), len(axis_B), len(axis_C)))
        for i in range(len(axis_A)) :
            for j in range(len(axis_B)) :
                for k in range(len(axis_C)) :
                    self.PDF [i, j, k] = (SM_xsec*PDF_SM[i, j, k] + BSM_xsec*PDF_BSM[i, j, k]) / tot_xsec

    print("sample and return")

    return tot_xsec, randomly_sample_grid (n_points, self.PDF, axis_A, axis_B, axis_C)

Simulator_Model2 = ParameterisedSimulator(name="Model2")
Simulator_Model2.add_param ("mu")
Simulator_Model2.set_function(generator_model2)


#  Brief: return {tot-xsec, datapoints}, with datapoints in the format {A, B, C}
#
def generator_model3 (self, n_points, *argv, **kwargs) :
    params    = {p.name:p.value for p in self.params}
    PoI       = params["c"]
    SM_xsec   = 2.
    BSM_xsec  = 0.3*PoI*PoI - 0.3*PoI 
    tot_xsec  = SM_xsec + BSM_xsec

    axis_A    = np.linspace(50, 250, 101)
    PDF_A_SM  = stats.cauchy.pdf(axis_A, 20, 70)
    PDF_A_SM  = PDF_A_SM / np.sum(PDF_A_SM)
    PDF_A_BSM = pow((axis_A / 200.),3) * stats.cauchy.pdf(axis_A, 20, 70)
    PDF_A_BSM = PDF_A_BSM / np.sum(PDF_A_BSM)
    PDF_A     = (SM_xsec*PDF_A_SM + BSM_xsec*PDF_A_BSM) / tot_xsec
    datapoints_A = randomly_sample_function (axis_A, PDF_A, n_points)
    #datapoints_A.sort()
            
    axis_B    = np.linspace(50, 500, 101)
    PDF_B_SM  = 1 / ((axis_B+300)*(axis_B+300)*(axis_B+300)*(axis_B+300))
    PDF_B_SM  = PDF_B_SM / np.sum(PDF_B_SM)
    PDF_B_BSM = stats.norm.pdf(axis_B, 250, 40)
    PDF_B_BSM = PDF_B_BSM / np.sum(PDF_B_BSM)
    PDF_B     = (SM_xsec*PDF_B_SM + BSM_xsec*PDF_B_BSM) / tot_xsec
    datapoints_B = randomly_sample_function (axis_B, PDF_B, n_points)
    #datapoints_B.sort()

    for i in range(n_points) :
        A = 50. + ((datapoints_A[i] - 50.) * (450./200.))
        B = datapoints_B[i]
        diff = A - B
        datapoints_B[i] = B + 0.5*diff


    axis_C    = np.linspace(-1.*np.pi, np.pi, 101)
    PDF_C_SM  = np.ones(axis_C.shape)
    PDF_C_SM  = PDF_C_SM / np.sum(PDF_C_SM)
    if PoI == 0 : PDF_C_BSM = np.ones(shape=axis_C.shape)
    else        : PDF_C_BSM = 1 + PoI * np.sin(axis_C) / np.fabs(PoI)
    PDF_C_BSM = PDF_C_BSM / np.sum(PDF_C_BSM)
    PDF_C     = (SM_xsec*PDF_C_SM + BSM_xsec*PDF_C_BSM) / tot_xsec
    datapoints_C = randomly_sample_function (axis_C, PDF_C, n_points)

    return tot_xsec, np.array([[a,b,c] for a,b,c in zip(datapoints_A, datapoints_B, datapoints_C)])

Simulator_Model3 = ParameterisedSimulator(name="Model3")
Simulator_Model3.add_param ("c")
Simulator_Model3.set_function(generator_model3)

