import numpy as np
from datetime import datetime
import random
from utils import *


def choquet_integral(data, bicap):
    """
    Computes choquet integral on data given a bicapacity

    Args:
        data (numpy array): MxN array of data 
        bicap (Bicapacity): bicapacity object 

    Returns:
        Mx1 numpy array of choquet integral fo each row
        
    """
    M, N = data.shape

    abs_data = np.abs(data)
    ind_sort = np.argsort(abs_data, axis=1)

    abs_data_sort = np.sort(abs_data, axis=1)
    diff_data = abs_data_sort.copy()
    diff_data[:, 1:] -= diff_data[:, :-1]

    data_sort = np.zeros_like(data)
    for r in range(data_sort.shape[0]):
        data_sort[r] = data[r][ind_sort[r]]

    ci = np.zeros((M, 1))
    for m in range(M):
        N_plus = ind_sort[m][data_sort[m] >= 0]
        N_minus = ind_sort[m][data_sort[m] < 0]
        for n in range(N):
            C = ind_sort[m, n:]

            N_plus_int_C = intersection(N_plus.tolist(), C.tolist())
            N_minus_int_C = intersection(N_minus.tolist(), C.tolist())

            ci[m, 0] += (
                    diff_data[m, n] * bicap.at(N_plus_int_C, N_minus_int_C, store_count=True)
                    )
    return ci


class Bicapacity:
    """
    Defines structure and usage of a bicapacity
    
    """
    def __init__(self, N, loadfile=None):
        """
        Constructor

        Args:
            N (int): number of sources 
            loadfile (str): path of load file (optional) 
        """
        self.N = N
        self.subsets = generate_subsets(N)
        self.subset_pairs = subset_pairs([i for i in range(N)])
        
        self.data = np.empty((2**N, 2**N))
        self.data[:, :] = np.nan
        self.bicap_count = np.zeros_like(self.data)

        #subset to index for faster lookup
        self.subset_to_ind = {}
        for i in range(len(self.subsets)):
            self.subset_to_ind[str(self.subsets[i])] = i
        
        #check to make sure it element can be set
        self.subset_pair_check = {}
        for pair in self.subset_pairs:
            self.subset_pair_check[str(pair)] = True

        if loadfile != None:
            self.load(loadfile)
    
    def ind_to_subset_pair(self, i, j):
        return (self.subsets[i], self.subsets[j])

    def subset_pair_to_ind(self, s1, s2):
        return (self.subset_to_ind[str(s1)], self.subset_to_ind[str(s2)])

    def at(self, s1, s2, set=None, store_count=False):
        """
        Gets bicapacity element at subset pair (s1,s2). If value is passed
        to set param, the element is at that pair is set to the value.
    
        Args:
            s1 (list): first subset 
            s2 (list): second subset 
            set (float): value to set at the (s1,s2) pair 
            store_count (bool): if true, store count for number of times that
                element is used

        Raises:
            Exception: invalid subset pair is provided

        Returns:
            Bicapacity element at subset pair (s1, s2)
            
        """
        if not str((s1,s2)) in self.subset_pair_check:
            raise Exception(f"Invalid bi-capacity subset pair {(s1,s2)}")
       
        ind1 = self.subset_to_ind[str(s1)]
        ind2 = self.subset_to_ind[str(s2)]
        
        if set != None:
            self.data[ind1][ind2] = set
        
        if store_count:
            self.bicap_count[ind1][ind2] += 1
        
        return self.data[ind1][ind2]
    
    def load(self, filename):
        """
        Load bicapacity from file        

        Args:
            filename (str): npy file for bicapacity 

        Raises:
            Exception: invalid path
        """
        if not os.path.exists(filename):
            raise Exception("Invalid filepath for bicapacity loading")
        self.data = np.load(filename)
        print(f"Loaded bicapacity: {filename}")

    def save(self, filename=None):
        """
        Save bicapacity to npy file        

        Args:
            filename (str): npy file path, if None, create filename with date 
        """
        if filename == None:
            now = datetime.now()
            filename = now.strftime("bicap_%Y%m%d_%H%M%S.npy")
        np.save(filename, self.data)
        print(f"Saved bicapacity: {filename}")

    def verify_monotonicity(self):
        """
        Verify if bicapacity is monotonic

        Returns:
            true if monotonic
            
        """
        for pair1 in self.subset_pairs:
            for pair2 in self.subset_pairs:
                if pair1 != pair2:
                    C, D = pair1
                    E, F = pair2
                    if is_subset_of(E, C) and is_subset_of(D, F):
                        val1 = self.at(C, D)
                        val2 = self.at(E, F)
                        if val1 != np.nan and val2 != np.nan:
                            if val1 < val2:
                                print(self)
                                print(f"{pair1}={val1} not >= {pair2}={val2}")
                                return False
        return True
   
    def __str__(self):
        np.set_printoptions(suppress=True)
        return np.array2string(self.data, precision=2)

class BicapacityGenerator:
    """
    Class to generate bicapacities
    """
    def __init__(self, N, useZeroBound=True):
        """
        Constructor

        Args:
            N (int): number of sources
            useZeroBound (bool): determines whether to enforce 0 bound (Obj 1 vs Obj 2) 
        """
        self.num_sources = N
        self.subset_pairs = subset_pairs([i for i in range(N)])
        self.useZeroBound = useZeroBound
        self._compute_bounds()

    def random_init(self, useZeroBound=True):
        """
        Randomly initialize a bicapacity        

        Args:
            useZeroBound (bool): enforce bicap zero bound if true 

        Returns:
            randomly generated Bicapacity object
            
        """
        bicap = Bicapacity(self.num_sources)
        full = [i for i in range(self.num_sources)]
        bicap.at([], full, set=-1.0)
        bicap.at(full, [], set=1.0)
        bicap.at([], [], set=0.0)

        queue = []
        queue.append(([],[]))

        set_pairs = []

        while len(queue) > 0:
            s1, s2 = queue.pop()
            set_pairs.append((s1, s2))
            
            ub = self.upper_bounds.at(s1, s2)
            lb = self.lower_bounds.at(s1, s2)

            if s1 == full and s2 == []:
                bicap.at(s1, s2, set=1.0)
            elif s1 == [] and s2 == full:
                bicap.at(s1, s2, set=-1.0) 
            elif s1 == [] and s2 == []:
                bicap.at(s1, s2, set=0.0) 
            else:
                min_upper = 1.0
                for u in ub:
                    if bicap.at(u[0], u[1]) < min_upper:
                        min_upper = bicap.at(u[0], u[1])
                max_lower = -1.0
                for l in lb:
                    if bicap.at(l[0], l[1]) > max_lower:
                        max_lower = bicap.at(l[0], l[1])

                val = random.uniform(max_lower, min_upper)
                bicap.at(s1, s2, set=val)

            coin = random.uniform(0, 1)
            if coin >= 0.5:
                for l in lb:
                    if l not in queue and l not in set_pairs:
                        queue.append(l)
                for u in ub:
                    if u not in queue and u not in set_pairs:
                        queue.append(u)
            else:
                for u in ub:
                    if u not in queue and u not in set_pairs:
                        queue.append(u)
                for l in lb:
                    if l not in queue and l not in set_pairs:
                        queue.append(l)
        return bicap 
    
    def _compute_bounds(self):
        self.lower_bounds = Bicapacity(self.num_sources)
        self.lower_bounds.data = self.lower_bounds.data.astype(object)
        self.upper_bounds = Bicapacity(self.num_sources)
        self.upper_bounds.data = self.upper_bounds.data.astype(object)

        for pair in self.subset_pairs:
            self.lower_bounds.at(pair[0], pair[1], set=[])
            self.upper_bounds.at(pair[0], pair[1], set=[])
        
        for p1 in self.subset_pairs:
            for p2 in self.subset_pairs:
                if p1 != p2:
                    C, D = p1
                    E, F = p2
                    if is_subset_of(E, C) and is_subset_of(D, F):
                        #C, D is upper bound of E, F
                        #E, F is lower bound of C, D
                        if self.useZeroBound:
                            self.lower_bounds.at(C, D).append(p2)
                            self.upper_bounds.at(E, F).append(p1)
                        else:
                            if C == [] and D == []:
                                self.lower_bounds.at(C, D).append(p2)
                            elif E == [] and F == []:
                                self.upper_bounds.at(E, F).append(p1)
                            else:
                                self.lower_bounds.at(C, D).append(p2)
                                self.upper_bounds.at(E, F).append(p1)

