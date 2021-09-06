# **Name Project**

## **About**

`RiskEvaluation`, is a Python package for creating a Risk Based Analysis Model. This package is based on customer segmentation, which involves categorizing the portfolio by industry, location, revenue, account size, and number of employees and many other variables to reveal where risk and opportunity live within the portfolio. Those patterns can then provide key measurable data points for more predictive credit risk management. Taking a portfolio approach to risk management gives credit professionals a better fix on the accounts, in order to develop strategies for better serving segments that present the best opportunities. Not only that, you can work to maximize performance in all customer segments, even seemingly risky segments.
Customer segmentation analysis can lead to several tangible improvements in credit risk management: stronger credit policies, and improved internal communication and cooperation across teams. 

## **How To**
`RiskEvaluation` package its comprehended by a Class called `RiskDataframe` The class is used to extend the properties of Dataframes to a prticular type of Dataframes in the Risk Indutry. 

It provides the end user with both general and specific cleaning functions, though they never reference a specific VARIABLE NAME.
    
It facilitates the End User to perform some Date Feature Engineering, Scaling, Encoding, etc. to avoid code repetition.



## Data Handling

`SetAttributes` function will update the type of the variable submitted for change. It will veify first that the key is present in the desired dataframe.

If present, it will try to change the type to the desired format. If not possible, it will continue to the next element.         

------------------------        
### Parameters

kwargs : The key-argument pair of field-type relationship that wants to be updated.

Returns

None.

------------------------

## Risk Based Approach

`missing_not_at_random` function will indicate if there is a relationship between missing data and its values.


## Data Cleaning

Once the user has indicated in Jupyter notebook which features are pivot_value, target_value, down_payment, income_status, birth_date and dates today. The `start` function will proceed to clean the dataset in order to be suitable to prepare it for the logistic regression model.

- `_pivot_unique` function will be used to remove duplicates based on the pivot value assigbned by the user.
- `_clean_target` function will convert the target values in a binary bucket, in order to reduce the complexity for the Logistic regression model.
- `_clean`  function will clean the empty spaces with 'UNKNOWN'.  
- `age` function will obtain the age from the birth date in the data frame, empty spaces will be filled with 'UNKNOWN'.
- `_down` function will obtain the down payment from eac, additionally the data frame is separated between individuals and corporate.
- `_income` function will simplify professions into unemployed or active in order to make professions an easier feature for segmentation.
- `_dayslapsed` function will convert days into numerical values which are more useful for the model.











































which perform the following actions:

1. Data Cleaning.
1. Duplicate removal, using a feature set by the user as pivot value.
1. Set our target as a binary variable.
1. Creating new features in order to obtain a better insight from the dataset such as:
	- Age
	- Days lapsed
	- Percentage of down payment
	- Income
1. Create a segmentation for categorical variables.
1. Create a segmentation for numerical variables.
1. Compare the accuracy of a logistic regression model with a full dataset vs. the segmented dataset.
1. Report generation in order to indicate which features are better to analyze with the segmented dataset or with the whole dataset.


## Example





## Please cite the package in publications!
By using `RiskEvaluation` you agree to the following rules:

You must place the following URL in a footnote to help others find `RiskEvaluation`: ~~https://CRAN.R-project.org/package=RiskPortfolios.~~

~~You assume all risk for the use of `RiskEvaluation`.
Ardia, D., Boudt, K., Gagnon-Fleury, J.-P. (2017a).
RiskPortfolios: Computation of risk-based portfolios in R.
Journal of Open Source Software, 10(2), 1.
https://doi.org/10.21105/joss.00171~~



