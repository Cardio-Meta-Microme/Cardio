### Use Cases
1. Run a ML model
a. Inputs: metabolome concentrations and microbiome relative abundance
b. Output: prediction of heart disease classification
2. Single user prediction:
a. Inputs: one patient’s metabolome and microbiome datasets. A user interface to easily add files.
b. Output: visualization of user’s risk score relative to training set
c. Datavis Ideas: Dimensionality reduction and clustering (mapping some outcome onto reduced dimension), cox proportional hazard, Regression Plots for Enrichments and disease.
3. Retraining ML model with additional data
a. Inputs: metabolome concentrations, microbiome relative abundances, and confirmed disease classification, number of iterations
b. Output: model (new set of feature weights), ROC plots
