# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
# New record
new_record = [[50, 9, 4, 1, 0, 0, 40, 0]]

# Code starts here
##Loading the dataset and saving it as a NumPy array
data = np.genfromtxt(path, delimiter = ",", skip_header = 1)

##Adding a fresh record to our dataset
census = np.concatenate([data, new_record], axis = 0)


# --------------
#Code starts here
##Creating a new array from the earlier census array
age = census[:, 0]

##Saving the maximum, minimum, mean, and standard deviation of the freshly created "age" array
max_age = age.max()
min_age = age.min()
age_mean = age.mean()
age_std = age.std()

##Printing out the above statistics to ascetain whther country is young or old
print(max_age, min_age, age_mean)

##Looking at the above, it seems that the country is old


# --------------
#Code starts here
##Creating four sub-arrays for each race
race_0 = census[census[ : , 2] == 0]
race_1 = census[census[ : , 2] == 1]
race_2 = census[census[ : , 2] == 2]
race_3 = census[census[ : , 2] == 3]
race_4 = census[census[ : , 2] == 4]

##Computing the length of each subarray and saving the same
len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)

##Creating a list of above lengths to find the smallest minority race
len_str = (len_0, len_1, len_2, len_3, len_4)

##Storing the "race code" of the subarray with minimum length as the smallest minority
minority_race = len_str.index(min(len_str))





# --------------
#Code starts here
##Creating a new subarray comprising senior citizens
senior_citizens = census[census[ : , 0] > 60]

##Outputting the sum of senior citizens' working hours and saving it as a variable
working_hours_sum = senior_citizens[ : , 6].sum()

##Outputting the length of the senior citizens subarray and saving it as a variable
senior_citizens_len = len(senior_citizens)

##Using the above variables to compute average working hours for senior citizens, and printing out the same
avg_working_hours = working_hours_sum/senior_citizens_len
print(avg_working_hours)

##Since average working hours for senior citizens is higher than 25, the government policy is not being followed 


# --------------
#Code starts here
##Creating two subae=rrays to capture two education-level categories
high = census[census[ : , 1] > 10]
low = census[census[ : , 1] <= 10]

##Obtaining the mean income for the two subarrays, and saving the same as variables
avg_pay_high = high[ : , 7].mean()
avg_pay_low = low[ : , 7].mean()

##Checking if higher education correlates with higher income
print(avg_pay_high > avg_pay_low)

##Given the output, yes, higher education does correlate with higher income


