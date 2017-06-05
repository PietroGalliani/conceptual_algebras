#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:58:03 2017

@author: Pietro Galliani

Some (very) simple functions to explore conceptual algebras.
"""

import numpy as np
from scipy import sparse
import pdb
import matplotlib.pyplot as plt
from numpy.random import binomial, choice, randint

def to_pairs(N, vals): 
    """
    Transform array of values from 0...N^2 to array of coordinates over a (N+1)x(N+1) matrix.
    The rows and column indices are always different from zero (that is, the input coordinates 
    are interpreted for the (1,1)-(N+1,N+1) submatrix)
    
    Used to generate the random matrix corresponding to the operation of the conceptual algebra. 
    
    
    Parameters
    ----------
    N : int 
        Number of elements in algebra (= height, width of output matrix)
    vals: 1-dimensional array of integers (0..N^2), length K 
        Positions of non-zero elements of matrix
        
    Returns
    ----------
    array (size K x 2)
        Positions of non-zero elements. 
        
        
    >>> to_pairs(2, np.arange(4))

    array([[ 1.,  1.],
           [ 2.,  1.],
           [ 1.,  2.],
           [ 2.,  2.]])
    
    """
    coords = np.empty((len(vals), 2))
    
    coords[:,0] = 1+ vals % N
    coords[:,1] = 1+ (vals - coords[:,0]+1)/N
    
    return coords


def random_op_matrix(N, density):
    """
    Generate random sparse NxN matrix of a given density
    
    The nonzero entries of the matrix are random values from 0 to N-1.
    
    Element 0 corresponds to the "undefined" concept: as such, column and row 
    0 do not contain nonzero values. 
    
    
    Parameters
    ----------
    
    N : int 
        Number of elements of conceptual algebra 
    density: float, (0..1)
        Probability that a given operation a . b is possible
        
    Returns 
    ----------
    
    csr sparse matrix (size N+1 x N+1)
        matrix of the partial algebra, in sparse form. 
        If the (i,j) entry is 0, then the outcome of i . j is undefined; 
        otherwise, the entry is the result is the operation. 
    
    >>> conceptual_algebras.random_op_matrix(5, 0.5).todense()
    matrix([[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4],
            [0, 5, 0, 0, 1, 0],
            [0, 4, 3, 2, 4, 3],
            [0, 5, 0, 5, 0, 0],
            [0, 0, 2, 0, 1, 4]], dtype=int64)
    """
    
    # Draw the number of nonzero (that is, not undefined)  operation results
    num_nonzero = binomial(N**2, density)
    
    # Pick random coordinates of nonzero elements
    coords = to_pairs(N, choice(N**2, num_nonzero,replace=False))
    
    # Choose the values of these elements (that is, the results of the
    #                                      corresponding concept combinations) 
    data = 1+randint(N, size=num_nonzero)
    
    # Generate sparse matrix
    r = sparse.csr_matrix((data, (coords[:,0], coords[:,1])), shape = (N+1, N+1))
    return r

class StepResult: 
    """ 
    Represents the results of one deduction - that is, a set of all the 
    known facts after the deduction, and the number of new facts that have
    just been learned.
    
    Just a simple container class, nothing to see here. 
    
    Arguments (= Attributes)
    ---------
    
    known_concepts: array
        Array of known concepts (1..N, where N is the number of elements)
            
    new_concepts: integer
        Number of novel concepts that have just been derived
        
    Attributes 
    ---------
    
    known_concepts: array
        Array of known concepts (1..N, where N is the number of elements)
            
    new_concepts: integer
        Number of novel concepts that have just been derived
        
    """
    def __init__(self, known_concepts, new_concepts):
        
        self.known_concepts = known_concepts
        self.new_concepts = new_concepts
        
class ConceptualSpace:
    """
    Represents a conceptual space. 
    
    Arguments 
    ---------
    
    num_concepts : int
        Number of concepts in conceptual algebra
    density: float (0..1)
        Probability that a given operation a . b is possible
    
    Attributes 
    ---------
    num_concepts : int
        Number of concepts in conceptual algebra
    density: float (0..1)
        Probability that a given operation a . b is possible
        
    op_matrix: csr sparse matrix (size num_concepts x num_concepts), values in 0..1
        op_matrix[i,j] = outcome of (i+1).(j+1), or 0 if undefined
    """
    
    def __init__(self, num_concepts, density=0.001):
        self.num_concepts = num_concepts
        self.density=density
        self.op_matrix = random_op_matrix(num_concepts, density)
        
    def one_step(self, theory, prev_results): 
        """
        Take an array of known concepts as inputs, perform one deduction step. 
        
        Arguments 
        ---------
    
        theory : 1-dimensional array of integers in 1..self.num_concepts 
            base concepts of our theory
        prev_results: StepResult
            results obtained in the previous steps
            
        Returns 
        ----------
        StepResult 
             concepts obtainable after one further step, and number of new concepts
        
        """
        
        
        known_new = np.unique(self.op_matrix[theory, :][:, prev_results.known_concepts].data)
        # Because of scipy's internal representation of sparse matrices, known_new will never contain 0
        
        known_after = np.unique(np.concatenate((known_new, prev_results.known_concepts))) 
        #inefficient, could probably find a better way
        
        return StepResult(known_after, known_after.size - prev_results.known_concepts.size)
    
    def spans_and_edges(self, theory, max_steps = 100):
        """
        Compute the spans and the edges corresponding to a theory. 
        
        Arguments
        ------- 
        theory : 1-dimensional array of integers (1..num_concepts) 
            Concepts in our theory
        max_steps: int (defalt 1000)
            Maximum number of derivation steps to consider
            
        Returns
        -------
        spans : array of integers
            spans[n] = <theory>_n = number of concepts derivable from theory within n steps
            
        edges : array of integers
            edges[n] = <theory>_n = number of new concepts that would be added at step n+1
        """
        r = StepResult(theory, theory.size)
        
        spans = np.zeros(max_steps)
        edges = np.zeros(max_steps)
        
        cur_step = 0
        spans[0] = theory.size
        while (r.new_concepts>0 and cur_step < max_steps): 
            r = self.one_step(theory, r)
            edges[cur_step]=r.new_concepts
            spans[cur_step+1] = spans[cur_step] + edges[cur_step]
            cur_step += 1
            
        if cur_step == max_steps: 
            print("Reached max step size!")
            pdb.set_trace()
        else:
            spans[cur_step:max_steps] = spans[cur_step]
            
        return spans, edges
    
    def avg_spans_and_edges(self, T, num_tries = 1000, max_steps=100):
        """
        Compute the average spans and edges of a theory of size T
        
        Arguments
        ---------
        
        T : int
            Number of elements in theory
            
        num_tries : int (default 1000)
            Number of theories on which to compute the average
            
        max_steps : int (default 100)
            Max number of derivation steps to consider
            
        Returns 
        ---------
        
        avg_spans : array of integers
            avg_spans[n] = E(<theory>_n) = avg. number of concepts derivable from theory within n steps
            
        avg_edges : array of integers
            avg_edges[n] = E(<theory>_n) = avg. number of new concepts that would be added at step n+1
            
        """
        avg_spans = np.zeros(max_steps)
        avg_edges = np.zeros(max_steps)
        
        for t in range(1,num_tries+1):
            known_concepts = 1+np.random.choice(self.num_concepts, T,replace=False) # Remember, 0 is the undefined concept       
            spans, edges = self.spans_and_edges(known_concepts, max_steps=max_steps)
            avg_spans += spans
            avg_edges += edges
            
        avg_spans/=num_tries
        avg_edges/=num_tries
        return avg_spans, avg_edges
    
    def compute_avg_value(self, T, C, L, num_tries=1000, max_steps=100):
        """
        Arguments
        --------
        
        T : int
            Number of elements in theory
        
        C : float OR multidimensional array of floats, positive
            Cost(s) per concept
            
        L : float, OR multidimensional array of floats, same shape as C, in 0..1
            Concept derivation discount(s) rate
            
        num_tries : int (default 1000)
            Number of theories on which to compute the average
            
        max_steps : int (default 100)
            Max number of derivation steps to consider
            
        Returns
        -------
        float (or array, same shape of C): 
            Avg value
            
        float (or array, same shape of C)
            Stddev value
            
        float: 
            Avg span. Note that this does not depend on C or L, so it is a single float in any case
        
        float: 
            Stddev span. Again, just a float
        """
        
        avg_values = np.zeros(np.shape(C))
        avg_span = 0
        
        # Squares of differences from current average, for online computation of stddevs
        # I want to avoid keeping the values for all samples in memory, because if 
        # C and L are big matrices that would consume a lot of memory.
        
        sqdiff_values = np.zeros(np.shape(C))
        sqdiff_span = 0
        
        factors = np.expand_dims(L, axis=len(np.shape(L)))**(np.arange(1, max_steps+1))
        
        for t in range(1,num_tries+1):
            known_concepts = 1+np.random.choice(self.num_concepts, T,replace=False) # Remember, 0 is the undefined concept       
            
            spans, edges = self.spans_and_edges(known_concepts, max_steps=max_steps)
            
            new_values = (1-C) * T + np.sum(factors*edges, axis = len(np.shape(C)))
            delta_values = (new_values - avg_values)
            avg_values += delta_values/t
            delta2_values = (new_values - avg_values)
            sqdiff_values += delta_values * delta2_values
            
            new_span = spans[-1]
            delta_span = (new_span - avg_span)
            avg_span += delta_span / t
            delta2_span = (new_span - avg_span)
            sqdiff_span += delta_span * delta2_span
            
        
        return avg_values, np.sqrt(sqdiff_values/(num_tries-1)), avg_span, np.sqrt(sqdiff_span/(num_tries-1))
    
    
    def compute_avg_value_2(self, T, C, L, num_tries=1000, max_steps=100):
        """
        Computes the average value of a theory of size T under concept cost C and discount rate L. 
        
        Differently from compute_avg_value, this function relies on avg_spans_and_edges; therefore, it is 
        not able to compute the standard deviation of the value. It might be a little faster, however, and 
        it is useful for debugging. 
        
        As before, C and L can also be arrays of values (same shape)
        
        Arguments
        --------
        
        T : int
            Number of elements in theory
        
        C : float OR multidimensional array of floats, positive
            Cost(s) per concept
            
        L : float, OR multidimensional array of floats, same shape as C, in 0..1
            Concept derivation discount(s) rate
            
        num_tries : int (default 1000)
            Number of theories on which to compute the average
            
        max_steps : int (default 100)
            Max number of derivation steps to consider
            
        Returns
        -------
        float (or array, same shape of C): 
            Avg value
            
        float: 
            Avg span. Note that this does not depend on C or L, so it is a single float in any case
        """
        avg_spans, avg_edges = self.avg_spans_and_edges(T, num_tries=num_tries,max_steps=max_steps)
        v = (1-C) * T
        factors = np.expand_dims(L, axis=len(np.shape(L)))**(np.arange(1, max_steps+1))
        v += np.sum(factors*avg_edges, axis = len(np.shape(C)))
        return v, avg_spans[-1]
    
    def best_size(self, min_C=0, max_C=5, step_C=0.005, min_L=0, max_L=1, step_L=0.005, step_size=100, max_d=100, num_tries=1000): 
        """
        Computes the statistics - avg value, stddev of value, avg span, stddev of span - for the optimal (highest avg value) 
        theory size, for different choices of concept cost C and discount rate L. 
        
        Arguments
        --------
        min_C : float (default 0) 
            Minimum value considered for concept cost C
        max_C : float (default 5)
            Max value considered for C
        step_C : float (default 0.005)
            Step length for concept cost
        min_L : float (default 0)
            Min value for discount rate L
        max_L : float (default 1)
            Max value for L 
        step_L : float(default 0.005)
            Steps length for L 
        step_size: int (default 100)
            Step length for theory size T
        max_d : int (default 100)
            Maximum derivation length for computing spans and edges
        num_tries: int (default 1000)
            Number of tries to average
            
        Returns
        --------
        BestSizeProfile
            Object containing the statistics about the best size for the various
            parameter choices, with saving and visualization methods. 
        """
        
        Cvals = np.arange(min_C, max_C+step_C, step_C)
        Lvals = np.arange(min_L, max_L+step_L, step_L)
    
        Cs, Ls = np.meshgrid(Cvals, Lvals, indexing='ij')
        
        sizes = np.zeros((len(Cvals), len(Lvals)))
        vals = -np.inf * np.ones((len(Cvals), len(Lvals)))
        vals_std = np.zeros(vals.shape)
        
        spans = np.zeros((len(Cvals), len(Lvals)))
        spans_std = np.zeros(spans.shape)
    
        for i in range(0, self.num_concepts+1, step_size): 
            
            print("N = %d"%i)
                
            
            v, v_std, s, s_std = self.compute_avg_value(i, Cs, Ls, num_tries = num_tries)
            #v, s = self.compute_best_value(i, Cs, Ls)
            best_changed = v > vals
            sizes[best_changed] = i
            vals[best_changed] = v[best_changed]
            vals_std[best_changed] = v_std[best_changed]
            spans[best_changed] = s
            spans_std[best_changed] = s_std
         
        return BestSizeProfile(Cvals, Lvals, sizes, vals, vals_std, spans, spans_std)
        #return Cs, Ls, sizes, vals, vals_std, spans, spans_std
    
class BestSizeProfile: 
    """
    Represents the optimal (greatest avg value) sizes for various choices of C and L, 
    together with the corresponding info about value, value stddev, span, span std. 
    
    Contains visualization methods and methods to save/load on disk. 
    
    
    Note: the constructor can be invoked with a filename (in which case it loads the data
    from the file) or by specifying Cvals, Lvals, ... spans_std, as in the 
    ConceptualSpace.BestSize function. 
    
    Arguments
    --------
    Cvals : array 
        Values for the C parameter
    Lvals : array
        Values for the L parameters
    sizes : array
        Optimal theory sizes
    values : array
        Optimal theory values
    values_std : array
        Standard deviations for optimal size theory values
    spans : array
        Total spans of optimal theories
    spans_std : array
        Standard deviations for optimal size theory spans
    filename : string
        name of the file from which to load the data
        
    Attributes
    --------
    Cvals : array 
        Values for the C parameter
    Lvals : array
        Values for the L parameter
    sizes : array
        Optimal theory sizes
    values : array
        Optimal theory values
    values_std : array
        Standard deviations for optimal size theory values
    spans : array
        Total spans of optimal theories
    spans_std : array
        Standard deviations for optimal size theory spans
        
    """
    
    def __init__(self, Cvals=None, Lvals=None, sizes=None, values=None, 
                 values_std=None, spans=None, spans_std=None, filename="best_size.npz"): 
        
        if (Cvals is None): 
            self.load_data(filename)
        else:
            self.Cvals = Cvals 
            self.Lvals = Lvals
            self.sizes = sizes
            self.values = values
            self.values_std = values_std
            self.spans = spans
            self.spans_std = spans_std
            
    
    def save_data(self, filename="best_size.npz"):
        """
        Saves profile in file. 
        
        Arguments
        ---------
        filename : string (default "best_size.npz") 
            Name of the file in which to save the data

        Returns
        ---------
        Nothing. 
        """
        np.savez(filename, **{"Cvals": self.Cvals, "Lvals":self.Lvals, "sizes": self.sizes, 
                              "values": self.values, "values_std":self.values_std, 
                              "spans":self.spans, "spans_std":self.spans_std})
    
    def load_data(self, filename="best_size.npz"):
        """
        Loads profile from file (usually invoked from constructor)

        Arguments
        ---------
        filename : string (default "best_size.npz") 
            Name of the file from which to load the data
        
        Returns
        --------
        Nothing
        """
        f = np.load(filename)
        
        self.Cvals = f["Cvals"] 
        self.Lvals = f["Lvals"]
        self.sizes = f["sizes"]
        self.values = f["values"]
        self.values_std = f["values_std"]
        self.spans = f["spans"]
        self.spans_std = f["spans_std"]


    def plot_phase_tr(self): 
        """
        Plots "phase transition diagrams" describing how the optimal theory sizes, their relative values and spans, and the corresponding standard deviations are affected by changes in C or L. 

        Arguments
        --------
        None 

        Returns 
        --------
        Nothing

        """
        CC, LL = np.meshgrid(self.Cvals, self.Lvals, indexing='ij')
        
        plt.figure()
        plt.pcolormesh(CC, LL, self.sizes)
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Derivation Discount Rate ($\lambda$)")
        plt.title(r"Optimal theory size")
        cb = plt.colorbar() 
        cb.set_label("Optimal theory size")
        
        
        
        plt.figure()
        plt.pcolormesh(CC, LL, self.values)
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Derivation Discount Rate ($\lambda$)")
        plt.title(r"Expected values for optimal size theories")
        cb = plt.colorbar() 
        cb.set_label("Expected theory value")
        
        plt.figure()
        plt.pcolormesh(CC, LL, self.values_std)
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Derivation Discount Rate ($\lambda$)")
        plt.title(r"Std of values for optimal size theories")
        cb = plt.colorbar() 
        cb.set_label("Theory value standard deviation")
        
        plt.figure()
        plt.pcolormesh(CC, LL, self.spans)
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Derivation Discount Rate ($\lambda$)")
        plt.title(r"Expected total span for optimal size theories")
        cb = plt.colorbar() 
        cb.set_label("Expected total span")
        
        plt.figure()
        plt.pcolormesh(CC, LL, self.spans_std)
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Derivation Discount Rate ($\lambda$)")
        plt.title(r"Std of total span for optimal size theories")
        cb = plt.colorbar() 
        cb.set_label("Total span standard deviation")
                
        
    def plot_Cs(self, start=0,step=50,end=200):
        """
        Plots expected theory value (for optimal theory size) as a function of concept cost C, for 
        some values of the discount rate L. 

        Arguments 
        ---------
        start : int (default 0) 
            Index of the first value self.Lvals[start] for which to plot theory value versus C
        step : int (default 50)
            Step size between different values of L considered 
        end : int (default 200)
            Index of the last value self.Lvals[end] for which to plot theory value versus C

        Returns 
        ---------
        Nothing
        """
        plt.figure()
        for i in range(start, end+1, step):
            plt.plot(self.Cvals, self.values[:,i], label=r"$\lambda$=%.2g"%self.Lvals[i])
            
        plt.xlabel(r"Concept Cost (C)")
        plt.ylabel(r"Expected value for optimal theory size")
        plt.title("Expected value as a function of concept cost")
        
        plt.legend()
        
    def plot_Ls(self, start=0,step=50,end=200):
        """
        Plots expected theory value (for optimal theory size) as a function of discount rate L, for 
        some values of the concept cost C. 

        Arguments 
        ---------
        start : int (default 0) 
            Index of the first value self.Cvals[start] for which to plot theory value versus L
        step : int (default 50)
            Step size between different values of C considered 
        end : int (default 200)
            Index of the last value self.Cvals[end] for which to plot theory value versus L

        Returns 
        ---------
        Nothing
        """

        plt.figure()
        for i in range(start, end+1, step):
            plt.plot(self.Lvals, self.values[i,:], label=r"C=%.2g"%self.Cvals[i])
            
        plt.xlabel(r"Discount Rate ($\lambda$)")
        plt.ylabel(r"Expected value for optimal theory size")
        plt.title("Expected value as a function of discount rate")
        
        plt.legend()
    
        
