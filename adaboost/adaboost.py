#################################
# Your name: Yoav Shoshan
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    hypotheses = []
    alpha_vals = []
    n = np.shape(X_train)[0]
    Dt = np.array([1/n]*n)
    for t in range(T):
        print('\nIteration: '+str(t))

        best_h_type, wl_j, theta, minimal_err, predictions = weak_learner(Dt,X_train,y_train)
        wt = 0.5*np.log((1-minimal_err)/minimal_err)
        ht = (best_h_type,wl_j,theta)
        hypotheses.append(ht)
        alpha_vals.append(wt)
        
        print(f'type:{ht[0]} index:{ht[1]} theta:{ht[2]} eps_t:{round(minimal_err,4)}')

        #updated_distribution
        factors = np.exp(-wt*np.array(predictions)*np.array(y_train))
        Dt = np.array([Dt[i]*factors[i] for i in range(n)])/np.dot(Dt,factors)
        
    return hypotheses, alpha_vals

##############################################

def calc_error(hypotheses,alpha_vals,x,y):
    n = np.shape(x)[0]
    err=0
    for xi, yi in zip(x,y):
        hypothesis_for_xi = 0 
        for alpha, h in zip(alpha_vals,hypotheses):
            if h[0] == 1:
                hxi = alpha if xi[h[1]]<=h[2] else -alpha
            else:
                hxi = -alpha if xi[h[1]]<=h[2] else alpha
            hypothesis_for_xi += hxi
        
        if np.sign(hypothesis_for_xi) != yi:
            err += 1/n
    return err
            

def calc_exponential_loss(hypotheses,alpha_vals,x,y):
    n = np.shape(x)[0]
    L=0
    for xi, yi in zip(x,y):
        predictions_on_xi = [alpha*ht[0] if xi[ht[1]]<=ht[2] else -alpha*ht[0] for ht, alpha in zip(hypotheses,alpha_vals)]
        L+=(1.0/n)*np.exp(-yi*sum(np.array(predictions_on_xi)))
        
    return L
    

def weak_learner(D, X_train, y_train):
    
    minimal_err = 1.0
    n = np.shape(X_train)[0]
    words_n = np.shape(X_train)[1]
    
    for j in range(words_n):

        words_count = X_train[:,j]
        words_count_set=sorted(list(set(words_count)))
        theta_arr=[words_count_set[0]-0.5] + [0.5*words_count_set[i]+0.5*words_count_set[i+1] for i in range(len(words_count_set)-1)] + [words_count_set[-1]+0.5]
        for cnt in set(theta_arr):
            
            h_plus_pred = [1 if x else -1 for x in words_count<=cnt]
            h_minus_pred = [-1 if x else 1 for x in words_count<=cnt]
            
            e_DSh_plus = sum([0 if h_plus_pred[i]==y_train[i] else D[i] for i in range(n)])
            e_DSh_minus = sum([0 if h_minus_pred[i]==y_train[i] else D[i] for i in range(n)])
            
            if min(e_DSh_plus,e_DSh_minus)<minimal_err:
                minimal_err = min(e_DSh_plus,e_DSh_minus)
                theta = cnt
                wl_j = j
                if e_DSh_plus<e_DSh_minus:
                    best_h_type = 1
                    predictions = h_plus_pred
                else:
                    best_h_type = -1
                    predictions = h_minus_pred
        
        if minimal_err == 1.0:
            print('All hypotheses are always wrong')
                
    return best_h_type, wl_j, theta, minimal_err, predictions


##############################################

def main():
    data = parse_data()
    if not data:
        return 
    (X_train, y_train, X_test, y_test, vocab) = data


    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    
    ##############################################
    #section a
    train_err = []
    test_err = []
    for t in range(T):
        train_err.append(calc_error(hypotheses[:t+1],alpha_vals[:t+1],X_train,y_train))
        test_err.append(calc_error(hypotheses[:t+1],alpha_vals[:t+1],X_test,y_test))
        
    plt.plot([t+1 for t in range(T)],train_err,marker='+')
    plt.plot([t+1 for t in range(T)],test_err,marker='+')
    plt.xlabel('t')
    plt.ylabel('Average Error')
    plt.legend(['Train Err', 'Test Err'], loc='best')
    plt.savefig('Error_wrt_t.png')
    plt.close()
    
    
    #section b
    print('\n')
    for i,h in enumerate(hypotheses[:10]):
        print(f'h{i}: word={vocab[h[1]]}: classify as {h[0]} if word occur <= {h[2]}')
    
    #section c
    exp_train_loss=[]
    exp_test_loss=[]
    
    for t in range(T):
        exp_train_loss.append(calc_exponential_loss(hypotheses[:t+1],alpha_vals[:t],X_train,y_train))
        exp_test_loss.append(calc_exponential_loss(hypotheses[:t+1],alpha_vals[:t],X_test,y_test))
        
    plt.plot([t+1 for t in range(T)],exp_train_loss,marker='+')
    plt.plot([t+1 for t in range(T)],exp_test_loss,marker='+')
    plt.xlabel('t')
    plt.ylabel('Exponential Loss')
    plt.legend(['Train Err', 'Test Err'], loc='best')
    plt.savefig('Exp_loss_wrt_t.png')
    plt.close()
    ##############################################

if __name__ == '__main__':
    main()
    
    
    
# =============================================================================
# def run_adaboost_2(X_train, y_train, T):
#     """
#     Returns: 
# 
#         hypotheses : 
#             A list of T tuples describing the hypotheses chosen by the algorithm. 
#             Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
#             the returned value (+1 or -1) if the count at index h_index is <= h_theta.
# 
#         alpha_vals : 
#             A list of T float values, which are the alpha values obtained in every 
#             iteration of the algorithm.
#     """
#     hypotheses = []
#     alpha_vals = []
#     n = np.shape(X_train)[0]
#     Dt = np.array([1/n]*n)
#     for t in range(T):
#         print('\n'+str(t))
# #        h_pred, h_index, h_theta, eps_t, predictions = weak_learner(Dt,X_train,y_train)
#         j,theta,eps_t = ERM_for_decision_stumps(Dt,X_train,y_train,1)
#         j2,theta2,eps_t2 = ERM_for_decision_stumps(Dt,X_train,y_train,-1)
# 
#         if eps_t<eps_t2:
#             h_pred = 1
#             ht = (h_pred,j,theta)
#             hyp_err = eps_t
#         else:
#             h_pred = -1
#             ht = (h_pred,j2,theta2)
#             hyp_err = eps_t2
#         
#         wt = 0.5*np.log((1-hyp_err)/hyp_err)
#         print(f'{ht[0]} {ht[1]} {ht[2]} {hyp_err}')
#         hypotheses.append(ht) 
#         alpha_vals.append(wt)
#         
#         #update distribution        
#         predictions = [ht[0] if x[ht[1]]<=ht[2] else -ht[0] for x in X_train]
#         factors = np.exp(-wt*np.array(predictions)*np.array(y_train))
#         Dt = np.array([Dt[i]*factors[i] for i in range(n)])/np.dot(Dt,factors)
# 
#     return hypotheses, alpha_vals
#    
#         
# def ERM_for_decision_stumps(D, X_train, y_train,pred_type):
#     """
#     this function finds the best j, and theta in a hypotheses class:
#         hj(x) = sign(theta-xj)
#     
#     we would like to minimize the objective:
#         
#     min_j min_theta (sum_on_i_yi_is_1(D_i*I(xi_j>theta) + sum_on_i_yi_is_-1(D_i*I(xi_j<=theta)))
#     see understanding-machine-learning-theory-algorithms 10.1.1
#     """
#     
#     F_star = float('inf')
#     n = np.shape(X_train)[0]
#     words_n = np.shape(X_train)[1]
#     
#     for j in range(words_n):
#         
#         words_count = X_train[:,j]
#         
#         #sort all arrays by xj's values
#         sorted_words_count = sorted(words_count)
#         sorted_words_count.append(sorted_words_count[-1]+1) #add one element that is slightly larger than the largest wordcount. in case best theta should be above all words_count
#         sorted_d = [d for _,d in sorted(zip(words_count,D))]
#         sorted_y = [y for _,y in sorted(zip(words_count,y_train))]
#         
#         #initialize objective with initial theta that is smaller that all words count
#         F = sum([sorted_d[i] if sorted_y[i]==pred_type else 0 for i in range(n)])
#         if F < F_star:
#             F_star = F
#             theta_star = sorted_words_count[0]-1
#             j_star = j
#                         
#         
#         for i in range(n):
#             F = F - pred_type*sorted_d[i]*sorted_y[i]
#             if F < F_star and sorted_words_count[i]!=sorted_words_count[i+1]:
#                 F_star = F
#                 theta_star = 0.5*(sorted_words_count[i]+sorted_words_count[i+1])
#                 j_star = j 
#     
#     #calculate error for weak learner
#     eps_t = 0
#     for i in range(n):
#         if (X_train[i][j_star]<=theta_star and y_train[i]==-pred_type) or (X_train[i][j_star]>theta_star and y_train[i]==pred_type):
#             eps_t+=D[i]
#             
#     return j_star,theta_star,eps_t
# =============================================================================
