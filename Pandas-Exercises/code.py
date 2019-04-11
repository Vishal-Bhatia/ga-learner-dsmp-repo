# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
##Loading the CSV file onto a Pandas DataFrame
bank = pd.read_csv(path)

##Creating a variable comprising information on categorical values, and printing the same
categorical_var = bank.select_dtypes(include = "object")
print(categorical_var)

##Creating a variable comprising information on numeric values, and printing the same
numerical_var = bank.select_dtypes(include = "number")
print(numerical_var)

# code ends here


# --------------
# code starts here


#code ends here
##Dropping the "Loan_ID" column
banks = bank.drop(["Loan_ID"], inplace = False, axis = 1)

##Checking for null values
print(banks.isnull().sum())

##Computing the mode
bank_mode = banks.mode

##Replacing Nas with mode
banks.fillna(bank_mode, inplace = True)

##Checking for null values
print(banks.isnull().sum())


# --------------
# Code starts here




# code ends here
##Creating a pivot table to check for loan amounts by gender, marital status, and employment status
avg_loan_amount = banks.pivot_table(index = ["Gender", "Married", "Self_Employed"], values = "LoanAmount", aggfunc = "mean")


# --------------
# code starts here





##Computing the number of self-employed with loans
loan_approved_se = len(banks[(banks["Self_Employed"] == "Yes") & (banks["Loan_Status"] == "Y")])

##Computing the number of non-self-employed with loans
loan_approved_nse = len(banks[(banks["Self_Employed"] == "No") & (banks["Loan_Status"] == "Y")])

##Computing the percentage of loan approvals where the borrower was self-employed
percentage_se = (loan_approved_se/614)*100

##Computing the percentage of loan approvals where the borrower was not self-employed
percentage_nse = (loan_approved_nse/614)*100


# code ends here


# --------------
# code starts here
##Creating a new column with loan tenure in years
loan_term = banks["Loan_Amount_Term"].apply(lambda x: x/12)

##Computing number of long-term (25 years or nore) loans and storing the same as a variable
big_loan_term = len(banks[banks["Loan_Amount_Term"] >= 300])




# code ends here


# --------------
# code starts here
##Grouping by loan status
loan_groupby = banks.groupby(banks["Loan_Status"])

##Subsetting applicant income and credit history columns
loan_groupby = loan_groupby[["ApplicantIncome", "Credit_History"]]

##Saving the mean of the subsetted dataframe as a variable
mean_values = loan_groupby.mean()

# code ends here


