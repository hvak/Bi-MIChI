from bicap import *
import random
#from scipy.stats import truncnorm
from multiprocessing import Pool

class BicapEvolutionaryTrain:
    """
    Bi-MIChI optimization framework
    """
    def __init__(self, Bags, Labels, Params):
        """

        Args:
            Bags (numpy array): 1xN data bags. Numpy array of numpy array objects 
            Labels (numpy array): size N array of bools 
            Params (dict): optimization params
                
        """
        self.Bags = Bags
        self.Labels = Labels
        self.Params = Params

        self.num_sources = self.Bags[0].shape[1]

        self.bicap_generator = BicapacityGenerator(self.num_sources, self.Params["use_zero_bound"])

    def small_mutation(self, bicap):
        counts = bicap.bicap_count.copy() 
        # ignore the boundaries 
        counts[-1, 0] = 0
        counts[0, -1] = 0
        counts[0, 0] = 0

        counts = counts.flatten()
        sum = counts.sum()
        probs = counts / sum

        select = np.random.multinomial(1, probs, size=1)
        ind = np.argmax(select)

        size = len(bicap.subsets)
        
        row = ind // size
        col = ind % size

        pair = bicap.ind_to_subset_pair(row, col)

        lb = self.bicap_generator.lower_bounds.at(pair[0], pair[1])
        ub = self.bicap_generator.upper_bounds.at(pair[0], pair[1])
        min_upper = 1.0
        for u in ub:
            if bicap.at(u[0], u[1]) < min_upper:
                min_upper = bicap.at(u[0], u[1])
        max_lower = -1.0
        for l in lb:
            if bicap.at(l[0], l[1]) > max_lower:
                max_lower = bicap.at(l[0], l[1])

        # UNIFORM 
        val = random.uniform(max_lower, min_upper)

        # TRUNCATED NORMAL
        #mean = bicap.at(pair[0], pair[1])
        #stdev = 0.1
        #a, b = (max_lower - mean) / stdev, (min_upper - mean) / stdev
        #print(a, b)
        #val = truncnorm.rvs(a, b)

        bicap.at(pair[0], pair[1], set=val)
        return bicap

    def large_mutation(self, bicap):
        return self.bicap_generator.random_init()

    def objective(self, bicap):
        pos_bags = self.Bags[self.Labels == 1]
        neg_bags = self.Bags[self.Labels == 0]

        num_pos = pos_bags.shape[0]
        num_neg = neg_bags.shape[0]

        obj_neg = 0
        for a in range(num_neg):
            ci = choquet_integral(neg_bags[a], bicap)
            if self.Params["use_zero_bound"]:
                obj_neg += ((ci + 0) ** 2).max(axis=1)[0]
            else:
                obj_neg += ((ci + 1) ** 2).max(axis=1)[0]

        obj_pos = 0
        for b in range(num_pos):
            ci = choquet_integral(pos_bags[b], bicap)
            if self.Params["use_zero_bound"]: 
                obj_pos += ((1 - np.abs(ci)) ** 2).min(axis=1)[0]
            else: 
                obj_pos += ((ci - 1) ** 2).min(axis=1)[0]

        return obj_neg + obj_pos

    def train(self, verbosity=1):
        """
        Run the optimization algorithm

        Args:
            verbosity (int): stride for printing iter 

        Returns:
            Bicapacity
            
        """
        print("PARAMS: ", self.Params)

        bicap_pop = [self.bicap_generator.random_init() for _ in range(self.Params["pop_size"])]
        obj_pop = [self.objective(bicap) for bicap in bicap_pop]

        obj_best = min(obj_pop)
        bicap_best = bicap_pop[np.argmin(obj_pop)]

        for i in range(self.Params["max_iter"]):
            #debug info
            if verbosity > -1:
                if i % verbosity == 0:
                    print("IT=", i)

            for p in range(self.Params["pop_size"]):
                z = random.uniform(0, 1)
                if z < self.Params["eta"]:
                    bicap_pop[p] = self.small_mutation(bicap_pop[p])
                else:
                    bicap_pop[p] = self.large_mutation(bicap_pop[p])
                obj_pop[p] = self.objective(bicap_pop[p])
            
            obj_best_new = min(obj_pop)
            obj_dist = abs(obj_best_new - obj_best)

            if obj_best_new < obj_best:
                print("FOUND IMPROVED MEASURE")
                obj_best = obj_best_new
                bicap_best = bicap_pop[np.argmin(obj_pop)]

            if obj_dist != 0 and obj_dist <= self.Params["fitness_thresh"]:
                print("REACHED STOPPING CRITERIA")
                break
          
        return bicap_best
    
    def train_multi(self, verbosity=1):
        pass
