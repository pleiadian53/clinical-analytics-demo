Case Study: Predicting Health Status Through Prescriptions and Diagnoses
========================================================================


Data Overview
-------------

In this case study, we will be looking at methods to define a patient’s health status using diagnosis data, and then try to predict a patient’s health status using prescription data only. This is a problem relevant in healthcare, for example as we try to understand a patient’s future therapeutic needs. 

For this task, we have available three datasets that simulate data for approximately 85,000 patients over a period of 3 years: 

1.	Patient Diagnosis (`Diagnosis.csv`). Each row of this dataset contains a patient’s diagnosis provided on a specific date. The diagnosis codes are presented in a standard called ICD-10.  

2.	Patient Prescriptions (`Prescriptions.csv`). Each row of this dataset contains a patient’s prescription filled on a specific date. The prescriptions contain drug category, drug group, and drug class.

3.	ICD-to-Clinical Categories Map (CCS). Each row in this file contains an ICD-10 diagnosis code (with a slightly different formatting than in the Patient Diagnosis table) and diagnosis descriptions as explained [here](https://www.hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp). Note that NOT every diagnosis code has a CSS code, so we will have to work around this.

Example Tasks
-------------

1.	Defining a Patient’s Health Status. For this task we will focus on defining a patient’s health status based on their diagnostic data. 
The CCS map will be useful for this. Hint: Use whatever information is appropriate in the given data sets to define a robust characterization of each patient’s health status. Ideally, the characterization should be useful for establishing something like “Patient 0123 has anemia and skin infection.”

2.	Predicting Heath Status using Prescription Data alone. In this task, we will infer a patient’s health status using only their prescription data. In particular, we will build a model that would allow us to potentially predict the health status of patients outside the ones provided. 

Example Solutions (Version 0)
-----------------------------

a. Solution Files 

   - `Case Study - Prescriptions vs Health Status.ipynb`: Solution notebook 
   - `Case-Study-tentative-solution-barnett.pdf`: Documentation for the solution. Note that the **solution notebook** includes both the demo code for essential modules and functions - and the thought process by which the aforementioned tasks are addressed. 
   - `Case-Study-v0-short-presentation.mov`: A short presentation for this demo. 

b. Modules 

   - `feature_extractor.py`: A module for feature extraction based on (simple) NLP methods. 
   - `data_pipeline.py`: Data preprocessing
   - `evaluate.py`: Model evalution
   - `icd_utils.py` 
   - `plot_utils.py` 
   - `utils.py`

Advanced Solutions (Todo)
-------------------------
   - Collaborative filtering 
   - Seq2seq



