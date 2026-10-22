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

## Approach



## Output Example 

```bash
Example of model output score.

```

## Future Considerations

- Verifying flagged dependency origins against a maintained vetted-source registry.
- Risk score for each agentic live session trace.
- A full GUI implementation. The current window is a launch screen only; analysis output prints to the console.

