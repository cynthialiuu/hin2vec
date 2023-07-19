import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import argparse
import csv
import math
import matplotlib.ticker as ticker


def create_data(sample_size, ones, zeros, all):
    # Can change this
    sample_size = sample_size

    # Randomly sample from data with label = 1
    o_sample_num = random.choices(range(0, 1923), k=sample_size)
    new_data = np.zeros((2*sample_size,200))
    new_data[0] = ones[o_sample_num[0]]
    o_sample_num.pop(0)
    k = 1
    # Populate new_data
    for i in o_sample_num:
        new_data[k] = ones[i]
        k = k + 1

    # Randomly sample from data with label = 0
    z_sample_num = random.choices(range(0, 1068564), k=sample_size)
    for i in z_sample_num:
        new_data[k] = zeros[i]
        k = k + 1

    new_labels = np.zeros(2*sample_size)
    for i in range(0, sample_size):
        new_labels[i] = 1
    

    X, y = new_data, new_labels
    seed = 7
    test_size = 0.10
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Tune hyperparameters
    nlearning_rate, nmax_depth, nmin_child_weight, ncolsample_bytree, nsubsample = tune_params(X_train, y_train, X_test, y_test)

    # Use tuned hyperparameters to construct final model
    pred = final_model(nlearning_rate, nmax_depth, nmin_child_weight, ncolsample_bytree, nsubsample, X, y, all)
    #HEllo what is going on
    # X_train, y_train, X_test, y_test,

    return(pred)
        



def tune_params(X_train, y_train, X_test, y_test):
    learning_rate = 0.1
    subsample = 1
    colsample_bytree = 1
    max_depth = 6
    min_child_weight = 1
    # Construct model and baseline testing_score with baseline parameters
    xgb_classifier = XGBClassifier(eta = learning_rate, 
                                        max_depth = max_depth, 
                                        min_child_weight=min_child_weight, 
                                        colsample_bytree = colsample_bytree, 
                                        subsample = subsample
                                        )
    xgb_classifier.fit(X_train, y_train)
    y_predicted = xgb_classifier.predict(X_test)
    # Get best accuracy score
    best_testing_score = accuracy_score(y_test, y_predicted)


    
    # print(best_testing_score)

    # First group of tuning (subsample, colsample_bytree)
    subsample_range = np.arange(0.2, 1, 0.2)
    colsample_range = np.arange(0, 1, 0.2)

    for sub in subsample_range:
        for col in colsample_range:
            xgb_classifier = XGBClassifier(eta = learning_rate, 
                                           max_depth = max_depth, 
                                           min_child_weight=min_child_weight, 
                                           colsample_bytree = col, 
                                           subsample = sub 
                                           )
            xgb_classifier.fit(X_train, y_train)
            y_predicted = xgb_classifier.predict(X_test)
            testing_score = accuracy_score(y_test, y_predicted)

            # print(testing_score)

            # If testing_score is improved, then hyperparameter is updated
            if (testing_score > best_testing_score):
                best_testing_score = testing_score
                subsample = sub
                colsample_bytree = col

    # print(subsample, colsample_bytree)

    # Second group (learning rate)
    learning_rate_range = np.arange(0.1, 1, 0.1)

    for lr in learning_rate_range:
        xgb_classifier = XGBClassifier(eta = lr, 
                                       max_depth = max_depth, 
                                       min_child_weight=min_child_weight, 
                                       colsample_bytree = colsample_bytree, 
                                       subsample = subsample)
        xgb_classifier.fit(X_train, y_train)
        y_predicted = xgb_classifier.predict(X_test)
        testing_score = accuracy_score(y_test, y_predicted)

        # print(testing_score)

        if (testing_score > best_testing_score):
            best_testing_score = testing_score
            learning_rate = lr

    # print(lr)
    print(learning_rate, max_depth, min_child_weight, colsample_bytree, subsample)

    return(learning_rate, max_depth, min_child_weight, colsample_bytree, subsample)




def final_model(learning_rate, max_depth, min_child_weight, colsample_bytree, subsample, X, y, all):
    #, X_train, y_train, X_test, y_test
    # Input final paramters
    seed = 7
    model = XGBClassifier(eta = learning_rate, 
                            max_depth = max_depth, 
                            min_child_weight=min_child_weight, 
                            colsample_bytree = colsample_bytree, 
                            subsample = subsample,
                            seed = seed)
    
    

    # Matrix used for probability scores (decimals between 0 and 1
    auroc_scores = np.array([])
    auprc_scores = np.array([])

    test_size_range = np.arange(0.1, 0.2, 0.02)

    for i in test_size_range:
        seed = 7
        test_size = i
        # Split data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        model.fit(X_train, y_train)
        y_pred_auroc = model.predict(X_test)
        # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_auroc)
        # fpr2, tpr2, _ = metrics.precision_recall_curve(y_test,  y_pred_auroc)
        auc = metrics.roc_auc_score(y_test, y_pred_auroc)
        auprc = metrics.average_precision_score(y_test, y_pred_auroc)
        auroc_scores = np.append(auroc_scores, auc)
        auprc_scores = np.append(auprc_scores, auprc)

    auroc_mean = np.mean(auroc_scores)
    auroc_std = np.std(auroc_scores)
    auprc_mean = np.mean(auprc_scores)
    auprc_std = np.std(auprc_scores)


    x_lab = ["AUROC", "AUPRC"]
    x_pos = np.arange(len(x_lab))
    CTEs = [auroc_mean, auprc_mean]
    error = [auroc_std, auprc_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, align='center', alpha=0.5, ecolor='black', capsize=10, width=0.4)
    for i in range(len(CTEs)):
        plt.text(i, CTEs[i], CTEs[i], ha = 'center')
    ax.set_ylabel('Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lab)
    tick_spacing = 0.1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.ylim(0.5, 1)
    ax.set_title('AUROC and AUPRC Scores for HIN2Vec')
    ax.yaxis.grid(True)


    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

    # #create ROC curve
    # plt.plot(fpr, tpr, label="AUC = "+str(auc))
    # plt.plot(fpr2, tpr2, label="AUPRC = "+str(auprc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc=4)
    # plt.show()

                
    # Matrix for accuracy score (0 or 1)
    # Can be commented out
    y_acc = model.predict(all)
    erm = len(y_acc)
    y_frame = np.zeros((erm))
    y_true = np.loadtxt('mat_drug_protein.txt')
    count = 0
    for i in range(0, 708):
        for j in range(0, 1512):
            y_frame[count] = y_true[i][j]
            count = count + 1
    print("Accuracy of Model::", accuracy_score(y_frame,y_acc))

    y_pred = model.predict_proba(all)

    return(y_pred)




def drug_input(drug_id, cutoff, sample_size):
    # Bool for whether drug was inputted as an DB number or drug name
    integer = False
    DB = False

    # This is if inputted 1-708 drug number
    integer = drug_id.isdigit()

    if (integer == True):
        drug_id = int(drug_id)

    #If input is DB####
    if (integer == False):
        if ((drug_id[0] == 'D') & (drug_id[1] == 'B')):
            drug_DB_num = drug_id
            DB = True

    target_id = []

    # DB to 1-708
    drug_num_to_num_dict = {}
    # 1-708 to DB
    num_to_drug_num_dict = {}
    # Drug name to 1-708
    drugname_to_drug_num_dict = {} 
    # 1-708 to drug name
    drug_num_to_drugname_dict = {}
    # 1-1512 to P
    protein_num_to_Pnum_dict = {}
    # p to protein name
    proteinP_to_name_dict = {}

    # Creates drug_num_to_num_dict and num_to_drug_num_dict
    with open('drug.txt', 'r') as file:
        i = 1
        for line in file:
            drug_num_to_num_dict[line] = i
            num_to_drug_num_dict[i] = line
            i = i + 1

    #Creates drugname_to_drug_num_dict and drug_num_to_drugname_dict
    with open('drug_dict_map.txt', 'r') as file2:
        file_contents = file2.read() 

        # Split the file contents into lines 
        lines = file_contents.split('\n') 
        # Iterate through the lines and populate the dictionary 
        for line in lines: 
            value, key = line.split(':') 
            drugname = key.strip()
            what = value.strip()
            drugname_to_drug_num_dict[drugname] = what
            drug_num_to_drugname_dict[what + '\n'] = drugname

    # If input is drug name, get drug DB number
    if ((integer == False) & (DB == False)):
        if (drugname_to_drug_num_dict.get(drug_id, False)):
            drug_DB_num = drugname_to_drug_num_dict[drug_id]
        else:
            print("Error: Drug not found")
            return

    # Finds the drug number 1-708 using DB#####
    if (integer == False):
        key = drug_DB_num + '\n'
        exists = drug_num_to_num_dict.get(key, False)
        if (exists):
            drug_id = drug_num_to_num_dict[drug_DB_num + '\n']
        else:
            print("Error: Drug not found")
            return
        
    # Creates protein_num_to_Pnum_dict
    with open('protein.txt', 'r') as file3:
        file_contents = file3.read() 
        lines = file_contents.split('\n') 
        # TODO: MIGHT NOT NEED ^
        i = 1
        for line in lines:
            protein_num_to_Pnum_dict[i] = line
            i = i + 1

    # Creates proteinP_to_name_dict
    with open('protein_dict_map.txt', 'r') as file4:
        file_contents = file4.read() 
        lines = file_contents.split('\n') 

        for line in lines: 
            key, value = line.split(':') 
            proteinP_to_name_dict[key.strip()] = value.strip() 
        
    # Loads dataset    
    ones = np.loadtxt('ones.txt')
    zeros = np.loadtxt('zeros.txt')
    all = np.loadtxt('all_mine.txt')

    predictions = create_data(sample_size, ones, zeros, all)

    # Iterate through row of that drug, finds which proteins have predictions above the cutoff
    # labels = np.loadtxt('mat_drug_protein.txt')
    # & (labels[drug_id - 1][i - starting_row ] == 0)

    starting_row = (drug_id - 1) * 1512
    for i in range(starting_row, starting_row + 1512):
        if ((predictions[i][1] >= cutoff)):
            pair = (predictions[i][1], i - starting_row + 1)
            target_id.append(pair)

    #Sorts so the largest predictions are first        
    target_id.sort(reverse=True)

    # Finds P number of predicted proteins
    protein_P_num = []

    for i in target_id:
        protein_P_num.append(protein_num_to_Pnum_dict[i[1]])

    # Using target_id protein, finds names to all proteins
    proteins = []

    for i in protein_P_num:
        proteins.append(proteinP_to_name_dict[i])


    # Prints predicted proteins for inputted drug
    # for i in range(0, len(proteins)):
    #     print(proteins[i], target_id[i][0])

    # for protein in target_id:
    #         print(protein[1], protein[0])
  

    str = num_to_drug_num_dict[drug_id]
    enter = drug_num_to_drugname_dict[str]
    filename = enter + '_protein_interaction.csv'

    with open(filename, 'w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(["Protein", "Probability"])
        for i in range(0, len(proteins)):
            writer.writerow([proteins[i], target_id[i][0]])

        
    
    # with open('good_drug_protein_prob.csv', 'w', newline='\n') as f:
    #     # f.write('Drug Protein Probability\n')
    #     row = 0
    #     for i in range(0, 708):
    #         for j in range(0, 1512):
    #             if ((predictions[row][1] > cutoff) & (labels[i][j] == 0)):
    #                 # dn = num_to_drug_num_dict[i + 1]
    #                 # name = drug_num_to_drugname_dict[dn]
    #                 writer = csv.writer(f)

    #                 dn = num_to_drug_num_dict[i + 1]
    #                 name = drug_num_to_drugname_dict[dn]

    #                 pn = protein_num_to_Pnum_dict[j + 1]
    #                 pname = proteinP_to_name_dict[pn]

    #                 writer.writerow([name, pname, str(predictions[row][1])])
    #                 # f.write(str(i + 1))
    #                 # f.write(' ')
    #                 # # pn = protein_num_to_Pnum_dict[j + 1]
    #                 # # pname = proteinP_to_name_dict[pn]
    #                 # f.write(str(j + 1))
    #                 # f.write(' ')
    #                 # f.write(str(predictions[row][1]))
    #                 # f.write('\n')                  
    #             row = row + 1


    # with open('good_drug_protein_prob_names.txt', 'w') as f:
    #     f.write('Drug Protein Probability\n')
    #     row = 0
    #     for i in range(0, 708):
    #         for j in range(0, 1512):
    #             if ((predictions[row][1] > cutoff) & (labels[i][j] == 0)):
    #                 dn = num_to_drug_num_dict[i + 1]
    #                 name = drug_num_to_drugname_dict[dn]
    #                 f.write(name)
    #                 f.write(' ')
    #                 pn = protein_num_to_Pnum_dict[j + 1]
    #                 pname = proteinP_to_name_dict[pn]
    #                 f.write(pname)
    #                 f.write(' ')
    #                 f.write(str(predictions[row][1]))
    #                 f.write('\n')                  
    #             row = row + 1




def protein_input(protein_id, cutoff, sample_size):
    # Bool for whether drug was inputted as an DB number or drug name

    # DB to 1-708
    drug_num_to_num_dict = {}
    # 1-708 to DB
    num_to_drug_num_dict = {}
    # Drug name to 1-708
    drugname_to_drug_num_dict = {} 
    # 1-708 to drug name
    drug_num_to_drugname_dict = {}
    # 1-1512 to P
    protein_num_to_Pnum_dict = {}
    protein_Pnum_to_num_dict = {}
    # p to protein name
    proteinP_to_name_dict = {}
    name_to_proteinP_dict = {}

    # Creates drug_num_to_num_dict and num_to_drug_num_dict
    with open('drug.txt', 'r') as file:
        i = 1
        for line in file:
            drug_num_to_num_dict[line] = i
            num_to_drug_num_dict[i] = line
            i = i + 1

    #Creates drugname_to_drug_num_dict and drug_num_to_drugname_dict
    with open('drug_dict_map.txt', 'r') as file2:
        file_contents = file2.read() 

        # Split the file contents into lines 
        lines = file_contents.split('\n') 
        # Iterate through the lines and populate the dictionary 
        for line in lines: 
            value, key = line.split(':') 
            drugname = key.strip()
            what = value.strip()
            drugname_to_drug_num_dict[drugname] = what
            drug_num_to_drugname_dict[what + '\n'] = drugname
        
    # Creates protein_num_to_Pnum_dict
    with open('protein.txt', 'r') as file3:
        file_contents = file3.read() 
        lines = file_contents.split('\n') 
        # TODO: MIGHT NOT NEED ^
        i = 1
        for line in lines:
            protein_num_to_Pnum_dict[i] = line
            protein_Pnum_to_num_dict[line] = i
            i = i + 1

    # Creates proteinP_to_name_dict
    with open('protein_dict_map.txt', 'r') as file4:
        file_contents = file4.read() 
        lines = file_contents.split('\n') 

        for line in lines: 
            key, value = line.split(':') 
            proteinP_to_name_dict[key.strip()] = value.strip() 
            name_to_proteinP_dict[value.strip()] = key.strip()


    #frina protein number 1-1512
    p_name = name_to_proteinP_dict[protein_id]
    p_num = protein_Pnum_to_num_dict[p_name]

        
    # Loads dataset    
    ones = np.loadtxt('ones.txt')
    zeros = np.loadtxt('zeros.txt')
    all = np.loadtxt('all_mine.txt')

    predictions = create_data(sample_size, ones, zeros, all)



    drug_pred = []

    # labels = np.loadtxt('mat_drug_protein.txt')
    #  & (labels[math.ceil(i / 1512) - 1][p_num - 1] == 0)

    starting_row = p_num - 1
    for i in range(starting_row, 1070496, 1512):
        if ((predictions[i][1] >= cutoff)):
            pair = (predictions[i][1], math.ceil(i / 1512))
            drug_pred.append(pair)

    #Sorts so the largest predictions are first        
    drug_pred.sort(reverse=True)


    drug_BDB = []

    for i in drug_pred:
        a = num_to_drug_num_dict[i[1]]
        b = drug_num_to_drugname_dict[a]
        drug_BDB.append(b)
    
    filename = protein_id + '_drug_interaction.csv'

    with open(filename, 'w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(["Drug", "Probability"])
        for i in range(0, len(drug_BDB)):
            writer.writerow([drug_BDB[i], drug_pred[i][0]])




# # Construct the argument parser
# ap = argparse.ArgumentParser()

# # Add the arguments to the parser
# ap.add_argument("-d", "--drug", required=True,
#    help="drug name")
# ap.add_argument("-c", "--cutoff", required=True,
#    help="cutoff score")
# ap.add_argument("-a", "--choice", required=True,
#    help="drug or protein")
# args = vars(ap.parse_args())


choice = input("Enter your input (drug or protein): ")
name = input("Enter the drug/protein name: ")
cutoff = float(input("Enter your cutoff: "))
sample_size = int(input("Enter your sample size: "))

word1 = "drug"
word2 = "protein"

if word1 in choice.lower():
    drug_input(name, cutoff, sample_size)

elif word2 in choice.lower():
    protein_input(name, cutoff, sample_size)
else:
    print("Error: No valid choice made")




# Calculate the sum
# name = args['drug']
# cutoff = float(args['cutoff'])
# choice = args['choice']