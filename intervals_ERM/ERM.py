#################################
# Your name: Yoav Shoshan
#################################
import os
import random
from math import sqrt, log
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.cm as cm
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    
    def __init__(self):
        self.dist_intervals = [(0,0.2),(0.4,0.6),(0.8,1)]
        self.p1 = 0.8 
        self.p2 = 0.1  #probabilities for y=1 given x is within/outside dist_intervals, respectively
        self.delta = 0.1 #confidence level requiered for section e (srm)
        self.cross_valdiation_data = 0.2

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the points where drawn from the distribution P.
        """
        
        x = sorted(np.random.uniform(size = m))
        y = []
        
        #for each value in x, draw y value according to the distribution: 
        #P(y=1) = 0.8 if x value is in intervals union. 
        #P(y=1) = 0.1 otherwise     
        for xi in x:
            if any(lower <= xi <= upper for (lower, upper) in self.dist_intervals):
                y.append(1) if np.random.uniform() <= 0.8 else y.append(0)
            else:
                y.append(1) if np.random.uniform() <= 0.1 else y.append(0)   
        points = np.column_stack((x, y)) 
        return points
    
        

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        print('\nDrawing sample intervals\n')
        
        points = self.sample_from_D(m)
        best_intervals, besterr = intervals.find_best_interval(points[:,0],points[:,1],k)
        o_path = os.getcwd()
        fig, ax = plt.subplots()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Dataset Distribution and Best Intervals')
        xcoords = [0,0.2,0.4,0.6,0.8,1]
        for xc in xcoords:
            plt.axvline(x=xc,linestyle = ':',)
        
        #draw best_intervals
        lines = [[(best_intervals[i][0],0.5),(best_intervals[i][1],0.5)] for i in range(len(best_intervals))]
        lc = mc.LineCollection(lines, colors=['y' for i in range(len(best_intervals))], linewidths=3)
        ax.add_collection(lc)
        
        plt.scatter(points[:,0], points[:,1],marker = '.',color = 'black')
        plt.ylim(-0.1,1.1)
        plt.xlim(-0.05,1.05)
        plt.xticks([0,0.2,0.4,0.6,0.8,1])
        plt.yticks([0,1])

        plt.savefig(o_path+'/Best_Intervals_on_train_data.pdf')
        print('Best_Intervals_on_train_data.pdf saved to CWD\n')
        


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        print('\nRunning experiment_m_range_erm\n')
        
        mean_true_errs_array = []
        mean_sample_errs_array = []
        m_arr = np.arange(m_first,m_last+step,step)
        
        for m in m_arr:
            print(f'Running {T} times for m = {m}:')
            T_true_err = []
            T_sample_err = []
            
            for i in range(T):
                points = self.sample_from_D(m)
                best_intervals, besterr = intervals.find_best_interval(points[:,0],points[:,1],k)
                T_sample_err.append(besterr/float(m))
                true_err = self.calc_true_error(best_intervals)             
                T_true_err.append(true_err)
            
            true_err_average = sum(T_true_err)/T
            empirical_err_average = sum(T_sample_err)/T
            mean_true_errs_array.append(true_err_average)
            mean_sample_errs_array.append(empirical_err_average)
            print(f'True_err = {round(true_err_average,2)}, Empirical_err = {round(empirical_err_average,2)}')
            
        o_path = os.getcwd()
        fig, ax = plt.subplots()
        plt.xlabel('M (sample size)')
        plt.ylabel('Error')
        plt.title(f'Errors wrt M')
               
        ax.plot(m_arr, mean_true_errs_array, c='blue', label='True Err')
        ax.plot(m_arr, mean_sample_errs_array, c='red', label='Empirical Err')
        
        minimal_true_err = self.calc_true_error(self.dist_intervals)
        plt.axhline(y=minimal_true_err,linestyle = ':',)
        plt.text(m_arr[0],minimal_true_err+0.01,f'Minimal True Error = {round(minimal_true_err,2)}')
        
        plt.ylim(0,max(mean_true_errs_array+mean_sample_errs_array)+0.05)
        plt.xlim(m_first,m_last)
        plt.xticks(np.arange(m_first,m_last+step,15))
        plt.yticks(np.arange(0,round(max(mean_true_errs_array+mean_sample_errs_array),1)+0.1,0.05))
        ax.legend()

        plt.savefig(o_path+'/Errors_wrt_M.pdf')
        print('\nErrors_wrt_M.pdf saved to CWD\n')
          
        return np.vstack([np.array(mean_sample_errs_array),np.array(mean_true_errs_array)])
        
                
        
    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        
        print('\nRunning experiment_k_range_erm\n')

        true_errs_array = []
        sample_errs_array = []
        points = self.sample_from_D(m)
        k_arr = np.arange(k_first,k_last+step,step)
        
        for k in k_arr:
            print(f'k={k}')
            best_intervals, besterr = intervals.find_best_interval(points[:,0],points[:,1],k)
            sample_errs_array.append(besterr/float(m))
            true_err = self.calc_true_error(best_intervals)          
            true_errs_array.append(true_err)
            print(f'true_err = {round(true_err,2)}, Empirical_err = {round(besterr/float(m),2)}')
        
        o_path = os.getcwd()
        fig, ax = plt.subplots()
        plt.xlabel('K (number of intervals)')
        plt.ylabel('Error')
        plt.title(f'Errors wrt K')
               
        k_sorted_by_err = [(err,k) for err,k in sorted(zip(true_errs_array,k_arr))]
        ax.plot(k_arr, true_errs_array, c='blue', label='True Err')
        ax.scatter(k_sorted_by_err[0][1], k_sorted_by_err[0][0], c='green', label='Minimal True Err',marker = 'D')
        ax.plot(k_arr, sample_errs_array, c='red', label='Empirical Err')
        plt.ylim(0,max(true_errs_array+sample_errs_array)+0.05)
        plt.xlim(k_first-1,k_last+1)
#        plt.xticks(np.arange(m_first,k_first,k_last+step,15))
#        plt.yticks(np.arange(0,round(max(mean_true_errs_array+mean_sample_errs_array),1)+0.1,0.05))
        ax.legend()

        plt.savefig(o_path+'/Errors_wrt_K.pdf')
        print('\nErrors_wrt_K.pdf saved to CWD\n')
        
        return k_sorted_by_err[0][1]
                
        

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        print('\nRunning experiment_k_range_srm\n')
        
        true_errs_array = []
        sample_errs_array = []
        penalties = []
        points = self.sample_from_D(m)
        k_arr = np.arange(k_first,k_last+step,step)
        
        for k in k_arr:
            print(f'k={k}')
            best_intervals, besterr = intervals.find_best_interval(points[:,0],points[:,1],k)
            sample_errs_array.append(besterr/float(m))
            true_err = self.calc_true_error(best_intervals)          
            true_errs_array.append(true_err) 
            penalty = sqrt( (8/m)*( 2*k + 2*log(m/k) + log(40) ) )  # according to VC and Sauer, Shlelah, Perles theorms. see written solutin for math, recall the VCdim = 2k
            penalties.append(penalty)
            print(f'true_err = {round(true_err,2)}, Empirical_err = {round(besterr/float(m),2)}, penalty = {round(penalty,2)}')

        sum_of_emp_err_and_penalty =  np.array(sample_errs_array) + np.array(penalties)
        
        o_path = os.getcwd()
        fig, ax = plt.subplots()
        plt.xlabel('K (number of intervals)')
        plt.ylabel('Error')
        plt.title(f'Errors and penalty wrt K')
               
        k_sorted_by_err_and_penalty = [(err,k) for err,k in sorted(zip(sum_of_emp_err_and_penalty,k_arr))]
        
        ax.plot(k_arr, true_errs_array, color='blue', label='True Err')
        ax.plot(k_arr, sample_errs_array, color='red', label='Empirical Err')
        ax.plot(k_arr, penalties, color='green', label='Penalty')
        ax.plot(k_arr, sum_of_emp_err_and_penalty, color='black', label='Empirical Err + Penalty')
        ax.legend()
        
        plt.savefig(o_path+'/Errors_and_penalty_wrt_K.pdf')
        print('\nErrors_and_penalty_wrt_K.pdf saved to CWD\n')
        
        return k_sorted_by_err_and_penalty[0][1]


    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
            """
            
        print('\nRunning cross_validation\n')
        k_arr = np.arange(1,11) #for each experiment, try values of k from 1 to 10
        best_k_arr = []
        
        o_path = os.getcwd()
        fig, ax = plt.subplots()
        plt.xlabel('K (number of intervals)')
        plt.ylabel('Hold Out Error')
        plt.title(f'Hold out Errors wrt K')
        colors = cm.rainbow(np.linspace(0, 1, T))
        
        holdout_n = int(round(m*self.cross_valdiation_data,0))
        points = self.sample_from_D(m)
        
        for i in range(T):  #run experiment T times
            print(f'Experiment {i+1}:')
            idx = random.sample(range(0, m), m)
            holdout = points[idx[:holdout_n],:]
            train = np.array(sorted(points[idx[holdout_n:],:], key = lambda entry: entry[0]))
            hold_out_err = []
            best_intervals_for_k = []
            
            for k in k_arr:
                print(f'K = {k}')
                best_intervals, besterr = intervals.find_best_interval(train[:,0],train[:,1],k)
                hold_out_err.append(self.validate_intervals_on_holdout(best_intervals,holdout)/len(holdout))
                best_intervals_for_k.append(best_intervals)
                print([[round(val, 4) for val in sublst] for sublst in best_intervals])
                print(f'Empirical error = {round(besterr/len(train),3)}, Hold_Out error = {round(hold_out_err[-1],3)}, True error = {round(self.calc_true_error(best_intervals),3)}')
                
            k_sorted_by_holdout_err = [(err,k) for err,k in sorted(zip(hold_out_err,k_arr))]
            best_intervals_sorted_by_holdout_err = [(err,best_inter) for err,best_inter in sorted(zip(hold_out_err,best_intervals_for_k))]
            best_k_arr.append(k_sorted_by_holdout_err[0]) 
            ax.plot(k_arr, hold_out_err, color=colors[i], label=f'Exp {i+1}')
            print(f'\nBest K is {k_sorted_by_holdout_err[0][1]} with holdout error of {round(k_sorted_by_holdout_err[0][0],3)}')
            
        best_k = max(set(best_k_arr), key=best_k_arr.count)
        print(f'\nBest K from {T} experiments is {k_sorted_by_holdout_err[0][1]}')

        ax.legend()
        plt.savefig(o_path+'/Hold_Out_err_wrt_K.pdf')
        print('\nHold_Out_err_wrt_K.pdf saved to CWD\n')
        
        return best_k
        

    def validate_intervals_on_holdout(self, best_intervals, holdout_set):
        """
        input: best intervals found by erm algorithm based on train data
               set of holdout points (and labels)
        output: number of errors best_intervals yields  on holdout set
        """
        errs = 0
        for point in holdout_set:
            if any([inter[0]<=point[0]<=inter[1] for inter in best_intervals]):
                if not point[1]:
                    errs+=1
            else:
                if point[1]:
                    errs+=1
        return errs
                    
                    
    def find_interscection(self,intervals_1,intervals_2):
        """
        input: two sets of disjoint intervals 
        output: set of disjoint intervals that are an intersection of the two unions
        """
        intersection = []
        for inter1 in intervals_1:
            for inter2 in intervals_2:
                if any([inter1[0]<=inter2[i]<=inter1[1] for i in range(len(inter2))]) or (inter2[0]<=inter1[0] and inter2[1]>=inter1[1]):
                    if inter2[0]>=inter1[0]:
                        if inter2[1]<=inter1[1]:
                            intersection.append([inter2[0],inter2[1]])
                        else:
                            intersection.append([inter2[0],inter1[1]])
                    else:
                        if inter1[1]<=inter2[1]:
                            intersection.append([inter1[0],inter1[1]])
                        else:
                            intersection.append([inter1[0],inter2[1]])
        return intersection
                
                
        
    def find_complementary_intervasls(self,intervals,lower=0,upper=1):
        """
        input: a set of disjoint intervals, lower (minimal value), upper (maximal value)
        output: a set of complementary intervals between start and end
        """
        comp_intervals = []
        if lower < intervals[0][0]:
            comp_intervals.append([lower,intervals[0][0]])
            
        for i in range(len(intervals)-1):
            comp_intervals.append([intervals[i][1],intervals[i+1][0]])
                
        if upper>intervals[-1][1]:
            comp_intervals.append([intervals[-1][1],upper])
        
        return comp_intervals
            
    
    def calc_true_error(self,best_intervals):
        """
        calculate true error given best intervals and dist_intervals
        dist_intervals are the intervals of which for x within them y = 1 in prob p1
        and for x outside them y = 1 in prob p2
        """
        p1 = self.p1
        p2 = self.p2
        #length of all possible logical intersections
        P_in_best_in_dist = sum([inter[1] - inter[0] for inter in self.find_interscection(self.dist_intervals,best_intervals)])
        P_in_best_not_in_dist = sum([inter[1] - inter[0] for inter in self.find_interscection(self.find_complementary_intervasls(self.dist_intervals),best_intervals)])
        P_not_in_best_in_dist = sum([inter[1] - inter[0] for inter in self.find_interscection(self.dist_intervals,self.find_complementary_intervasls(best_intervals))])
        P_not_in_best_not_in_dist = sum([inter[1] - inter[0] for inter in self.find_interscection(self.find_complementary_intervasls(self.dist_intervals),self.find_complementary_intervasls(best_intervals))])
        
        #denote L = dist_intervals
        #and h(x)=1 ---> x in best_intervals, then
        #e_P(h) = P(Y=0|h(x)=1 and x in L)*P(h(x)=1 and x in L) + P(Y=0|h(x)=1 and x not in L)*P(h(x)=1 and x not in L)
        #        +P(Y=1|h(x)=0 and x in L)*P(h(x)=0 and x in L) + P(Y=1|h(x)=0 and x not in L)*P(h(x)=0 and x not in L)
        # and for example P(Y=1|h(x)=0 and x in L) = P(Y=1|x in L) = p1
        return ((1-p1)*P_in_best_in_dist + (1-p2)*P_in_best_not_in_dist) + (p1*P_not_in_best_in_dist + p2*P_not_in_best_not_in_dist )



if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)