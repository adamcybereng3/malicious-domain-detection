# Applying Machine Learning to Detect Malicious DGA Domains

This project uses the publicly available dga_domain dataset to detect malicious domains created by domain generations algorithms (DGAs). Regular and LASSO logistic regression are trained on the dataset. Both approaches are tested on families held out from training.

## Dataset

The dataset is known as the [DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset), available on GitHub. It was produced as part of a study from the Expert Systems and Applications Journal. The authors permitted public use of the data for research purposes. The domains were extracted from 25 DGA families from the Netlab Opendata Project and Alexa was were the benign domains were sourced from. The dataset was downloaded 9/2026 to support training the algorithms associated with this project. 

### Citation and DOI
```bash
Cucchiarelli, A., Morbidoni, C., Spalazzi, L., Baldi, M., Algorithmically generated malicious domain names detection based on n-grams features (2021) Expert Systems with Applications, 170, art. no. 114551
https://doi.org/10.1016/j.eswa.2020.114551
```
To use the dataset, it must be downloaded into a local code repository. Preferably, it should be stored in the same folder where the training and testing source code is housed. The dga_domains_full.csv file must be in the same folder as the script.

The file includes a series of three columns with no header. The three general categories include the label, family, and domain string. 

## Requirements

The script was tested on Python version 3.13.14. 

Multiple packages are used for data preprocessing, training/testing, and reporting. At minimum, it is recommended to install the following packages to ensure the script can have the necessary dependencies to execute all libraries. See the following command to install the appropriate packages:

```bash
python -m pip install pandas numpy scikit-learn matplotlib
```

## Running the Project

Ensure you are in the script's working directory in an IDE, command-line interface, or terminal. The following is the command to execute the script:

```bash
python evaluation.py
```
### Output

The script produces a combination of textual and graphical outputs. On the console, a series of tables will be printed which include a summary of the data's attributes, classification reports, LASSO coefficient scores, recall for every family, and base-rate results. Two separate plots will generate towards the end of the program, which include an ROC and Precision-Recall curve. The script pauses when each plot windows appears, and continues when the user closes each one.

Since all random steps are done on a fixed seed (random_state=0), the results will always be identitcal every run. 

The following are sample outputs from the final run:

#### LASSO Coefficients
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

#### Logistic Regression Seen vs. Unseen Families
```bash
===== Logistic regression - seen families =====
              precision    recall  f1-score   support

           0       0.83      0.87      0.85    101202
           1       0.69      0.61      0.65     45900

    accuracy                           0.79    147102
   macro avg       0.76      0.74      0.75    147102
weighted avg       0.79      0.79      0.79    147102

===== Logistic regression - unseen families =====
              precision    recall  f1-score   support

           0       0.68      0.87      0.77    101202
           1       0.84      0.62      0.72    108000

    accuracy                           0.74    209202
   macro avg       0.76      0.75      0.74    209202
weighted avg       0.77      0.74      0.74    209202
```




