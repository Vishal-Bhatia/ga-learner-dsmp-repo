# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Code starts here
##Loading the CSV file onto a Pandas DataFrame
data = pd.read_csv(path)

##Computing value-counts for loan status, and outputting the same as a bar graph
loan_status = data["Loan_Status"].value_counts()
loan_status.plot(kind = "bar")




# --------------
#Code starts here
##Creating the groupby dataframe by Property_Area and Loan_Status
property_and_loan = data.groupby(["Property_Area", "Loan_Status"])

##Applying the .size and .unstack methods
property_and_loan = property_and_loan.size().unstack()

##Plotting the graph as instructed
property_and_loan.plot(kind = "bar", stacked = False, figsize = (15,10))
plt.xlabel("Property Area", rotation = 45)
plt.ylabel("Loan Status")



# --------------
#Code starts here
##Creating the groupby dataframe by Education and Loan_Status
education_and_loan = pd.DataFrame(data.groupby(['Education', 'Loan_Status'])['Loan_Status'].size().unstack())

##Plotting the graph as instructed
education_and_loan.plot(kind = "bar", stacked = True, figsize = (15,10))
plt.xlabel("Education Status")
plt.ylabel("Loan Status")
plt.xticks(rotation = 45)

###5. Graduate
###6. Graduate


# --------------
#Code starts here
##Creating sliced datframes as requested
graduate = data[data["Education"] == "Graduate"]
not_graduate = data[data["Education"] == "Not Graduate"]

##Plotting density plots
graduate.plot(kind = "density", label = "Graduate")
not_graduate.plot(kind = "density", label = "Graduate")

#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
##Initializing the figure and Axes
fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3, ncols = 1)

##PLotting the scatter plot between ApplicantIncome and LoanAmount
ax_1 = data.plot.scatter(x = "ApplicantIncome", y = "LoanAmount")
ax_1.set_title("Applicant Income")

##PLotting the scatter plot between CoapplicantIncome and LoanAmount
ax_2 = data.plot.scatter(x = "CoapplicantIncome", y = "LoanAmount")
ax_2.set_title("Coapplicant Income")

##Creating a new TotalIncome column
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

##PLotting the scatter plot between TotalIncome and LoanAmount
ax_3 = data.plot.scatter(x = "TotalIncome", y = "LoanAmount")
ax_3.set_title("Total Income")

###7. 
###8. 



