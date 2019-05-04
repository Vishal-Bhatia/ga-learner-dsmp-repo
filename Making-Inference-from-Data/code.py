# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
##Loading the file onto a dataframe
data = pd.read_csv(path)

##Cutting a sample with random seed set at 0
data_sample = data.sample(n = sample_size, random_state = 0)

##Saving the sample mean and STD for "installment"
sample_mean = data_sample["installment"].mean()
sample_std = data_sample["installment"].std()

##Computing the margin of error, and then the confidence interval
margin_of_error = z_critical*(sample_std/math.sqrt(sample_size))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

##Computing the population mean for "installment"
true_mean = data["installment"].mean()

##Checking if the population mean lies in the computed confidence interval
print((true_mean >= sample_mean - margin_of_error) or (true_mean <= sample_mean - margin_of_error))





# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size = np.array([20, 50, 100])

#Code starts here
##Setting the figure plot area
fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (6, 6))

##Running a for loop for the three sample sizes suggested
for i in range(len(sample_size)):
    ##Initializing the empty list
    m = []
    ##Running a for loop that will range up to 1,000
    for j in range(1000):
        ##Appending above empty list with means, essentially taking a mean 1,000 times
        m.append(data["installment"].sample(sample_size[i]).mean())
    ##Converting the list to a pandas series
    mean_series = pd.Series(m)
    ##Plotting the histograms as separate subplots
    axes[i].hist(mean_series)
    


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
##Converting the strings in interest rate column into float objects
data["int.rate"] = data["int.rate"].apply(lambda x: float(x[0:len(x) - 1])/100)

##Applying the ztest
z_statistic, p_value = ztest(data[data["purpose"] == "small_business"]["int.rate"], value = data["int.rate"].mean(), alternative = "larger")

##Checking for the p-value
if p_value < 0.05:
    print("We reject the null hypothesis.")
else:
    print("We do not reject the null hypothesis.")


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
##Applying the two-sided Z test
z_statistic, p_value = ztest(data[data["paid.back.loan"] == "No"]["installment"], data[data["paid.back.loan"] == "Yes"]["installment"])

##Checking for the p-value
if p_value < 0.05:
    print("We reject the null hypothesis.")
else:
    print("We do not reject the null hypothesis.")


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
##Storing the value counts in variables as instructed
yes = data[data["paid.back.loan"] == "Yes"]["purpose"].value_counts()
no = data[data["paid.back.loan"] == "No"]["purpose"].value_counts()
type(yes.transpose())

##Concatanating the variables as instructed
observed = pd.concat((yes.transpose(), no.transpose()), axis = 1, keys = ["Yes", "No"])

##Applying the Chi-squared distribution 
chi2, p, dof, ex = chi2_contingency(observed)

##Comparing the chi2 and critical values
if chi2 < critical_value:
    print("We reject the null hypothesis.")
else:
    print("We do not reject the null hypothesis.")


