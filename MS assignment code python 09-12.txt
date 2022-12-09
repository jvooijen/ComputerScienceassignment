import math
from scipy.special import comb
import os
import json
import pandas as pd
import numpy as np
from time import perf_counter
import random
from random import shuffle
#from math import comb #deze nog installeren 
import re
import itertools
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

with open('C:/Users/jvanooijen/OneDrive - Deloitte (O365D)/Documents/Personal/Study/TVs-all-merged.json') as f:
    data = json.load(f) 


################################# Create dataframe ###################################    
minData = []
for k,v in data.items(): #for loop over dictionary of productID
    for element in v: #for loop over over list (items)
        title = element['title'] #Identify the titles
        shop = element['shop'] #Identify the shops
            
        features_dict = element['featuresMap']          
        features_string = ""
        
        for item in element['featuresMap'].values():
            features_string += item 
            features_string += " "  
                
            if('Brand' in element['featuresMap'].keys()):  #Identify the Television brands (if available)
                brand = element['featuresMap']['Brand']
            else: brand = None                    #brand is none if the brand is not available                    
            
        minData.append((k, title, shop, brand, features_string, features_dict)) #extract title from item
      
complete_data = pd.DataFrame(minData, columns = ['ID', 'Title', 'Shop', 'Brand', 'Features', 'FeaturesDict'])  


###################################### Data Cleaning #########################################

def cleandata(complete_data):
    
    for columns in ['Title']:    
        #set titles to lowercase
        complete_data[columns] = complete_data[columns].map(lambda title: title.lower()) 
         #remove shop titles as it can be assumed there are no duplicates within shops
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('newegg.com', '')) 
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('best buy', ''))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('thenerds.net', ''))
        #remove special signs from the set
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('-', '')) 
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace(',', ''))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace(';', ''))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('/', ''))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('.', ''))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace(':', ''))
        #replace dubble spatie by single spatie
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('  ', ' '))
        #in order to stay consistent, write 'hertz' in the same way
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('hertz', 'hz')) 
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('-hz', 'hz'))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace(' hz', 'hz'))
        #in order to stay consistent, write #inch in the same way
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('inches', 'inch')) 
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('"', 'inch'))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace('-inch', 'inch'))
        complete_data[columns] = complete_data[columns].map(lambda title: title.replace(' inch', 'inch'))
        complete_data[columns] = complete_data[columns].map(lambda title: re.sub('\W+', ' ', title))
       
    return complete_data
complete_data = cleandata(complete_data)  #completely cleaned dataset
complete_data2 = cleandata(complete_data) #use later 


############################ create vector of model words #############################

def modelwords(complete_data):

    regex = "([a-zA-Z0-9](([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9])" #as defined in the literature

    modelwords_array = [set() for _ in range(len(complete_data))]
    complete_modelwords = set() #create a set for the modelwords
    for i, row in complete_data.iterrows():                
        modelwords_title = set(itertools.chain.from_iterable(re.findall(regex, row['Title'])))   #In the row 'title;' identify modelwords, by using the regex 
        
        modelwords = modelwords_title
        complete_modelwords = complete_modelwords.union(modelwords)  
    
        modelwords_array[i] = modelwords 

    complete_modelwords = list(complete_modelwords)    
    complete_data['ModelWords'] = modelwords_array
    
    complete_data.loc[:, 'Binary_MW'] = complete_data['ModelWords'].map(lambda mw: [1 if a in mw else 0 for a in complete_modelwords]) #Make a binary frame of the modelwords; 1 if the modelwords is in the title, 0 if not

    return complete_data, complete_modelwords



############################## function that predict duplicates ##########################
def pred_duplicates(complete_data, n_band, n_row, complete_modelwords): 
    threshold = (1/n_band)**(1/n_row) #set threshold t (from the literature)
    print(f'Threshold = {round(threshold,2)} where #b = {n_band} and #r = {n_row}') #the row is the band width
    
    complete_data, complete_modelwords = modelwords(complete_data) 
    complete_data = add_sign(complete_data, n_band, n_row, complete_modelwords)   
    
    global counter #returns the dictionary of current global counter and buckets
    global buckets
    
    buckets = []  #create the buckets
    counter = 0
     
    #by minhashing into buckets we want to find the potential duplicates. I.e. items with corresponding modelwords belong to the same bucket and shoudl therefore be candidate duplicates.                             
                         
    for sig in complete_data['Signature']:    
        hash_sign(sig, n_band, n_row)

    for b in range(n_band):
        buckets.append({})

    candidates = derive_candidates(buckets) 
    comparisons_total = comb(len(complete_data),2)    #total number of comparisons
    comparisons_fraction = round(len(candidates)/comparisons_total,6) #fraction of comparisons

    print(f'Fraction of comparisons: {comparisons_fraction}')
    
    
    d_matrix = dist_matrix(complete_data, candidates) #returns dissimilarity matrix
    clusters = clustering(d_matrix, best_t[counter][i]) #uses dissimaliry matrix to create clusters
    duplicates = predict_duplicates(clusters) #returns the predicted duplicates from the clusters
    
    evaluations = list(evaluate(candidates, duplicates, complete_data))
    evaluations.append(comparisons_fraction)

    return evaluations

################################## Minhashing ####################################
# In this step we create a signature matrix of the binary vectors by using minhashfunctions  
#function signature, creates signatures based on binary vectors optimized 
def signature(mh_functions,v):
    index = np.nonzero(v)[0].tolist()  #appends non-zero array to list
    row_numbers = mh_functions[:, index] #define the rownumbers of 
    signature = np.min(row_numbers, axis=1)
    return signature

#minhash function, 
def minhash_func(len_mw, n_signature): 
    hashes = []
    for _ in range(n_signature):
        hash_vector = list(range(1, 1 + len_mw))
        shuffle(hash_vector)
        hashes.append(hash_vector)
    return hashes
    
#function add_signatures
def add_sign(complete_data, n_band, n_row, complete_modelwords):
    len_mw = 1563 #length of all the modelwords
    n_signature = n_band * n_row # n = b*r
    mh_functions = np.array(minhash_func(len_mw, n_signature)) 
    complete_data.loc[:,'Signature'] = complete_data['Binary_MW'].map(lambda binary: signature(mh_functions, binary))
    return complete_data 
  
    
################################### LSH #######################################

#hash signature function, hashes sub signatures into buckets
def hash_sign(signature, n_band, n_row):
    global buckets    
    global counter
    sub_signatures = np.array(split_sign(signature, n_band, n_row)).astype(object)
    sub_signs = str(sub_signatures)
    
    for i,subsignature in enumerate(sub_signs):  
        subsignature = ', '.join(subsignature)
        if subsignature not in buckets[i].keys():  
            buckets[i][subsignature] = []
        buckets[i][subsignature].append(counter)
    counter = counter + 1

#In this step we define the functions used to perform Locality Sensitive Hashing. 
#function that returns potential duplicates 
def derive_candidates(buckets):
    candidates = []
    for band in buckets:
        keys = band.keys()
        for bucket in keys:
            potential_duplicates = band[bucket] #this returns the array to the corresponding band 
            if len(potential_duplicates) > 1: 
                candidates.extend(combinations(potential_duplicates, 2))
    return list(set(candidates))

# split signature function takes signature matrix, number of bands and number of rows as input, creates sub signatures
def split_sign(signature, n_band, n_row):
    sub_signatures = []   
    for i in range(0, len(signature), n_band):
        sub_signatures.append(signature[i : i+n_row])      #create the sub signatures
    return sub_signatures  


# Jaccard similarity
def jaccard(x, y):    
    x = set(x)
    y = set(y)
    return float(len(x.intersection(y)) / len(x.union(y)))  


############################### Dissimilarity matrix ##############################
#function that creates a dissimilarity matrix that is used to perform the clustering 
def dist_matrix(complete_data, candidates): 
        
    d_matrix = np.full((len(complete_data), len(complete_data)), 1000.00)  

    for pair in candidates:
        
        if (complete_data.loc[pair[0], 'Shop'] == complete_data.loc[pair[1], 'Shop']):
            d_matrix[pair[0]][pair[1]] = 1000.00
            d_matrix[pair[1]][pair[0]] = 1000.00
        elif (complete_data.loc[pair[0],'Brand'] != None and complete_data.loc[pair[1],'Brand'] != None and complete_data.loc[pair[0],'Brand'] != complete_data.loc[pair[1],'Brand']):
            d_matrix[pair[0]][pair[1]] = 1000.00
            d_matrix[pair[1]][pair[0]] = 1000.00
        else: 
            d_matrix[pair[0]][pair[1]] = 1-jaccard(complete_data.loc[pair[0], 'Signature'], complete_data.loc[pair[1], 'Signature'])
            d_matrix[pair[1]][pair[0]] = 1-jaccard(complete_data.loc[pair[0], 'Signature'], complete_data.loc[pair[1], 'Signature'])

    return d_matrix
 

##################################  Clustering  ####################################
# with this function we create clusters by using average linkage clustering
def clustering(d_matrix, t):
    clusters = AgglomerativeClustering(affinity='precomputed', linkage='Complete', distance_threshold=t, n_clusters=None)
    clusters.fit(d_matrix)
    return clusters

def predict_duplicates(clusters): 
    duplicates =[]
    for cluster in range(clusters.n_clusters_): 
        content_products = np.where(clusters.labels_ == cluster)[0]
        if (len(content_products) > 1):
            duplicates.extend(list(combinations(content_products, 2)))
    return duplicates


################################### Evaluation #######################################
def evaluate(candidates, duplicates, complete_data):
    true_dupl = set() #create empty set
    complete_duplicates = 0 #total number of true duplicates
    
    for model_ID in complete_data['ID']: #compare model ID's to find true duplicates
        if model_ID not in true_dupl: 
            duplicates_ID = len(list(combinations(np.where(complete_data['ID'] == model_ID)[0], 2)))    
            complete_duplicates += duplicates_ID
            true_dupl.add(model_ID)
            
    candidate_TP = 0
    candidate_FP = 0            
    TP = 0
    FP = 0
    FN = 0

    predicted_duplicates = set(duplicates)
    
    for i in range(len(predicted_duplicates)):
        if (complete_data['ID'][duplicates[i][0]] == complete_data['ID'][duplicates[i][1]]):
            TP += 1
        else:
            FP += 1
   
    for i in range(len(candidates)):
        if (complete_data['ID'][candidates[i][0]] == complete_data['ID'][candidates[i][1]]):
            candidate_TP += 1
        else:
            candidate_FP += 1    
    
    number_comparisons = len(candidates)  
    #pairwise Quality     
    PQ = candidate_TP/number_comparisons #precision, ratio between true candidates and total comparisons
    #pairwise Completeness
    PC = candidate_TP/complete_duplicates #recall, ratio between true candidates and true duplicates
    
    FN = complete_duplicates - TP #false negatives
    F1_star = 2*((PQ * PC)/(PQ + PC)) #formula as given in the literature
    F1 = TP /(TP + (FP + FN)/2)

    return PQ, PC, F1, F1_star


#candidates = derive_candidates(buckets)
#d_matrix = dist_matrix(complete_data, candidates)

################################# BOOTSTRAP ##################################
n_bootstrap = 5 #minimum bootstraps is 5 in order to get a robust solution
percentage_bootstrap = 0.63 #63% as train data
#chosen pairs
b_r = [ [2, 750],  
        [30, 50], 
        [50, 30], 
        [150, 10], 
        [300, 5], 
        [500, 3],
        [750, 2]
            ]

t_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #range of threshold value

#####################################################  Bootstrap function ##########################
def bootstrap(complete_data, n_bootstrap, percentage_bootstrap, t_list):
    
    results = np.empty((1,5)) #create array of 1x5
    
    bt_obs = round(percentage_bootstrap)*len(complete_data) #63% of the dataframe
    best_t = np.zeros((len(b_r),n_bootstrap))
    
    for i in range(n_bootstrap): #for loop over the bootstraps
        print(f'Bootstrap {i+1}')

        train = np.random.choice(len(complete_data), bt_obs, replace = False) 
        complete_data_test = complete_data.drop(train, axis = 0)        
        complete_data_train = complete_data.iloc[train]
        complete_data_train.reset_index(inplace = True)
        complete_data_test.reset_index(inplace = True)
        
        pairs = 0
    
        for pair in b_r: #for loop over every pair 
            n_band = pair[0] 
            n_row = pair[1] 
            F1_max = -1 
            F1_star = 0
            PQ_sum = 0
            PC_sum = 0
            for t in t_list: #forloop over every threshold 
                evaluations = pred_duplicates(complete_data_train, n_band, n_row, t) 
                PQ_eval = evaluations[0]
                PC_eval = evaluations[1]
                F1_eval = evaluations[2]
                F1_star_eval = evaluations[3]
              
                print(f'Variables: {[n_band, n_row, t]}')
                if (F1_eval > F1_max):
                    F1_max = F1_eval
                    best_t[pairs][i] = t
                PQ_sum += PQ_eval
                PC_sum += PC_eval
                F1_star += F1_star_eval
            
            PC_eval = PC_sum/len(t_list)    
            PQ_eval = PQ_sum/len(t_list)
            F1_star_eval = F1_star/len(t_list)  
      
    results = np.append(results, evaluations, 0)
    pairs += 1

    return outsample_results[1:], best_t

     
outsample_results, best_t = bootstrap(complete_data2, n_bootstrap, percentage_bootstrap, t_list)        

#### print results
print(f'Nr of bootstraps: {n_bootstrap}')
print(f'b,r pairs: {b_r}')
print(f'Dist Thresholds: {t_list}')
    
    
average_bt_results = np.empty((len(b_r),outsample_results.shape[1]))
for i in range(len(b_r)):
    average_bt_results[i] = np.mean(outsample_results[i::len(b_r)],axis=0)

print(f'Max out of Sample F1: {np.max(average_bt_results[:,2])} (avg over {n_bootstrap} bootstraps)')
print(f'Max out of Sample F1*: {np.max(average_bt_results[:,3])} (avg over {n_bootstrap} bootstraps)')


###################################3#Plotting the results ######################################


#Pairwise comparison
plt.plot(average_bt_results[:,4], average_bt_results[:,0], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PQ")
plt.title("Out of Sample: Pairwise Quality vs Fraction of comparisons")
plt.show()

#Pairwise completeness
plt.plot(average_bt_results[:,4], average_bt_results[:,1], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("PC")
plt.title("Out of Sample: Pairwise Completeness vs Fraction of comparisons")
plt.show()

#F1* 
plt.plot(average_bt_results[:,4], average_bt_results[:,3], '-o')        
plt.axis([0, 0.2, 0, 0.2])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1*")
plt.title("Out of Sample: F1* vs Fraction of comparisons")
plt.show()

#F1 
plt.plot(average_bt_results[:,4], average_bt_results[:,2], '-o')        
plt.axis([0, 1, 0, 1])
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1")
plt.title("Out of Sample: F1 vs Fraction of comparisons")
plt.show()