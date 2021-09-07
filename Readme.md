# **Jigglypuff**

## **About**

`Jigglypuff`, is a Python package for creating a Risk Based Analysis Model. This package is based on customer segmentation, which involves categorizing the portfolio by industry, location, revenue, account size, and number of employees and many other variables to reveal where risk and opportunity live within the portfolio. Those patterns can then provide key measurable data points for more predictive credit risk management. Taking a portfolio approach to risk management gives credit professionals a better fix on the accounts, in order to develop strategies for better serving segments that present the best opportunities. Not only that, you can work to maximize performance in all customer segments, even seemingly risky segments.
Customer segmentation analysis can lead to several tangible improvements in credit risk management: stronger credit policies, and improved internal communication and cooperation across teams. 

## **How To**
`Jigglypuff` package its comprehended by a Class called `RiskDataframe` The class is used to extend the properties of Dataframes to a particular type of Dataframes in the Risk Indutry. 

It provides the end user with both general and specific cleaning functions, though they never reference a specific VARIABLE NAME.
    
It facilitates the End User to perform some Date Feature Engineering, Scaling, Encoding, etc. to avoid code repetition.

This package works setting as a target the column where are recorded the missing payments, this is our target and the prediction of the Logistic Regression model. 

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

## Data Analysis

### Categorical Variable Analysis

`RiskDataframe` class will automatically segment the categorical values in order to indicate if the split generates a better accuracy than the whole dataset. For categorical values it has been indicated a train of 10% and a test for the 90% of the data frame.

`seg_data_cat` is a list of columns which shall be specified by the user, in order to indicate the class which columns contains categorical values.

The method `set_train_cat(target_value,seg_data_cat)` will measure the effectiveness of splitting the categorical values of the data set.


### Encoding

Prior to execute the numerical variable analysis, it has to be converted the categorical values to numerical values. In order to make this, `encod` function will run one hot encoder.

### Numerical Variable Analysis

`RiskDataframe` class will automatically segment the numerical values in order to indicate if the split generates a better accuracy than the whole dataset. For cnumerical values it has been indicated a train of 10% and a test for the 90% of the data frame.

`seg_data_num` is a list of columns which shall be specified by the user, in order to indicate the class which columns contains numerical values.

The method `set_train_num(target_value,seg_data_num)` will measure the effectiveness of splitting the cnumerical values of the data set.

For both categorical and numerical values, the effectiveness is measured by GINI score.

## Example





## Please cite the package in publications!
By using `Jigglypuff` you agree to the following rules:

You must place the following URL in a footnote to help others find `Jigglypuff`: ~~https://CRAN.R-project.org/package=RiskPortfolios.~~

~~You assume all risk for the use of `Jigglypuff`.
Ardia, D., Boudt, K., Gagnon-Fleury, J.-P. (2017a).
RiskPortfolios: Computation of risk-based portfolios in R.
Journal of Open Source Software, 10(2), 1.
https://doi.org/10.21105/joss.00171~~



