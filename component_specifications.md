Front end 
User Interface
What it does: GUI that the user interacts with
Input: Interactions / clicks from user.
Outputs: Data visualization displayed to the user
Interacts with the back end and the data interface 

# Back end
### 1. Data Processing:
- What it does: Drop NAs, filter sparse data, remove irrelevant data, merge dfs, data transformation depending on distribution of data (e.g. log transform) 
- Inputs: Validated dataframe(s) (from Data Interface)
- Outputs: Processed dataframe (for model)
- How to use w/ other components: gets data from interface, sends processed data to model

### 2. Model:
- What it does: Random forest regression or classification
- Inputs: cleaned dataframe (from Data processing)
- Outputs: Predictions / classification in dataframe format with additional model stats/info (like regression summary) 
- How to use w/ other components: takes processed data, runs scikitlearn models, returns parameter dataframe and other necessary info in csv format for download

### 3. Data Visualization:
- What it does: plots model results, ROC plots 
- Inputs: model results and processed data
- Outputs: plot(s) in .jpg format? Interactive would be cool if possible
- How to use w/ other components: 


# Database and Data Interface
### 1. Raw Data Upload
- What it does: Exposes an API that allows for uploading of raw data
- Input: CSV files of datasets
- Output: Dataset saved to database
- Doesn’t directly interact with other components
### 2. Raw Data Download
- What it does: Exposes an API that allows for downloading of raw data
- Input: definitions of the datasets requested
- Output: CSV files of datasets
### 3. Load Data from External Source
- What it does: Accesses data from a external database
- Input: url to an external database
- Output: Data from external database
### 4. Validate User Input Data
- Check that the input data are valid for the visualizations.
- Inputs: One or more csv files containing patient cohort data
- Outputs: Tells the user if the input is valid or not
- Gives suggestions as to why the input doesn’t work (have some expected format etc)
- Feeds into the data processing module.
- Interacts with the front end, receives data from the user. Also passes data to the back end. 

# Documentation
- README.md file for an overview of the function of the project and tool.
- Input: Doc strings and descriptive, well styled code
- Output: Readable project
- Interacts with the backend, front end, and database and data interface

