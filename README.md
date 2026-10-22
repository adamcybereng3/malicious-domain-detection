# Applying Machine Learning to Detect Malicious DGA Domains

This project uses the publicly available dga_domain dataset to detect malicious domains created by the Domain Generation Algorithm (DGA). LASSO logistic regression and a decision tree are trained on the dataset. Both approaches are tested on families held out from training.

## Dataset

The dataset is known as the [DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset), available on GitHub. It was produced as part of a study from the Expert Systems and Applications Journal. The authors permitted public use of the data for research purposes. The dataset was downloaded 9/2026 to support training the algorithms associated with this project. 

To use the dataset, it must be downloaded and uploaded into a local code repository. Preferably, it should be stored in the same folder where the training and testing source code is housed. The following is the file fourmat:

```bash
dga_domains_full.csv
```
The file includes a series of three columns with no header. The three general categories include the label, family, and domain string. 

## Requirements

The script was run Python version 3.13.14. Ensure the Python version installed on your system is at least 3.13.

Multiple packages are used for data preprocessing, training/testing, and reporting. At minimum, it is reccomended to install the following packages to ensure the script can have the necessary dependencies to execute all libraries. See the following command to install the appropiate packages:

```bash
python -m pip install pandas numpy scikit-learn matplotlib
```

## Running the Project

Ensure you are in the script's working directory in an IDE, command-line interface, or terminal. The following is the command to execute the script:

```bash
evaluation.py
```
### Output

The script produces a combination of textual and graphical outputs. On the console, a series of tables will be printed which include a summary of the data's attributes, classification reports, LASSO coefficient scores, decision tree rules, recall for every family, and base-rate results. Two separate plots will be generate towards the end of the program, which include and ROC and Precision-Recall curve.

Since all random steps are done on a fixed seeed (random_state=0), the results will always be identitcal every run. 

#### LAMBDA Coefficients
```bash
Chosen lambda: 1291.549665014884
             feature  coefficient
5       hyphen_count    -0.340900
6  unique_char_ratio     0.000000
3            entropy     0.110783
1        digit_ratio     0.170117
2        vowel_ratio     0.170879
0             length     0.275170
4  max_consonant_run     1.576274
```

#### Decision Tree Seen vs. Unseen Families
```bash
===== Decision tree - seen families =====
              precision    recall  f1-score   support

           0       0.82      0.96      0.89    101202
           1       0.86      0.55      0.67     45900

    accuracy                           0.83    147102
   macro avg       0.84      0.75      0.78    147102
weighted avg       0.83      0.83      0.82    147102

===== Decision tree - unseen families =====
              precision    recall  f1-score   support

           0       0.60      0.96      0.74    101202
           1       0.92      0.41      0.57    108000

    accuracy                           0.68    209202
   macro avg       0.76      0.68      0.65    209202
weighted avg       0.76      0.68      0.65    209202
```




