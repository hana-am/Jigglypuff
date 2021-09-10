# **Jigglypuff**

## **About**

`Jigglypuff`, is a Python package for creating a Risk Based Analysis Model. This package is based on customer segmentation, which involves categorizing the portfolio by industry, location, revenue, account size, and number of employees and many other variables to reveal where risk and opportunity live within the portfolio. Those patterns can then provide key measurable data points for more predictive credit risk management. Taking a portfolio approach to risk management gives credit professionals a better fix on the accounts, in order to develop strategies for better serving segments that present the best opportunities. Not only that, you can work to maximize performance in all customer segments, even seemingly risky segments.
Customer segmentation analysis can lead to several tangible improvements in credit risk management: stronger credit policies, and improved internal communication and cooperation across teams. 

## **How To**
`Jigglypuff` package its comprehended by a Class called `RiskDataframe` The class is used to extend the properties of Dataframes to a particular type of Dataframes in the Risk Indutry. 

It provides the end user with both general and specific cleaning functions, though they never reference a specific VARIABLE NAME.
    
It facilitates the End User to perform some Date Feature Engineering, Scaling, Encoding, etc. to avoid code repetition.

This package works setting as a target the column where are recorded the missing payments, this is our target and the prediction of the Logistic Regression model. 

`Jigglypuff` can be pip installed, running the following command `pip install Jigglypuff`  

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

### 1. Using Autoloans.csv dataset
#### Library Installation
	!pip install Jigglypuff==0.2
	from Jigglypuff.RiskDataframe import RiskDataframe as rdf

	import pandas as pd

	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier

#### Name the dataset

	dataframe = pd.read_csv("AUTO_LOANS_DATA.csv", sep=";")
	myrdf = rdf.RiskDataframe(dataframe)

#### Implement method .missing_not_at_random()

	myrdf.missing_not_at_random(input_vars=[]) 

#### Variable indication

The user has to set the pivot column (Loand contract number), the target value (Unpaid monthly installments), down payment, income status or profession of the lender, birth date and the rest of the dates on the dataset.

	pivot_value = 'ACCOUNT_NUMBER'
	target_value = 'BUCKET'
	down_payment = 'PROGRAM_NAME'
	income_status = 'PROFESSION'
	birth_date = 'BIRTH_DATE'
	dates_todays = ['REPORTING_DATE','LOAN_OPEN_DATE','EXPECTED_CLOSE_DATE','CUSTOMER_OPEN_DATE']

	myrdf.start(pivot_value,birth_date,target_value, down_payment,income_status,dates_todays)

#### Setting the data types

In order to set the data types a dictionary has to be created indicating which are the categorical variables.

	argument_dict = {'PROGRAM_NAME':'category','SEX':'category',
                'PROFESSION':'category','CAR_TYPE':'category'}
	myrdf.SetAttributes(argument_dict)
	myrdf.dtypes


#### Categorical varibles indication
The user has to indicate which are the categorical variables from the dataset.

	seg_data_cat =['SEX','PROFESSION','CAR_TYPE','TYPE'] 

#### Run the model for categorical values

	myrdf.set_train_cat(target_value,seg_data_cat)

The output of this method will indicate the GINI score for the full model and the segmented model, for the categorical variables, and will be indicated below whether the split model is better or worse than the full model.

	(['The total accuracy using all variable and Logistic regression is: 0.8686868686868687',
  	'Using: SEX GINI Full Model Seg1: 30.303745053332285%',
  	'Using: SEX GINI Segmented Model Seg1: 30.303745053332285%',
  	'Using: SEX GINI Full Model Seg2: 19.219512195121947%',
  	'Using: SEX GINI Segmented Model Seg2:19.219512195121947%',
  	'Using: PROFESSION GINI Full Model Seg1: 26.49165350778253%',
  	'Using: PROFESSION GINI Segmented Model Seg1: 26.49165350778253%',
  	'Using: PROFESSION GINI Full Model Seg2: nan%',
  	'Using: PROFESSION GINI Segmented Model Seg2:nan%',
  	'Using: CAR_TYPE GINI Full Model Seg1: 47.844112769485925%',
  	'Using: CAR_TYPE GINI Segmented Model Seg1: 47.844112769485925%',
  	'Using: CAR_TYPE GINI Full Model Seg2: 22.12240785828228%',
  	'Using: CAR_TYPE GINI Segmented Model Seg2:22.12240785828228%',
  	'Using: TYPE GINI Full Model Seg1: 24.75394321766562%',
  	'Using: TYPE GINI Segmented Model Seg1: 24.75394321766562%',
  	'Using: TYPE GINI Full Model Seg2: 55.55555555555558%',
  	'Using: TYPE GINI Segmented Model Seg2:55.55555555555558%'],
 	['After analysis, we did not find a good split using: SEX',
 	 'After analysis, we find a good split using: PROFESSION set at: ACTIVE',
 	 'After analysis, we find a good split using: CAR_TYPE set at: HYUNDAI',
 	 'After analysis, we find a good split using: TYPE set at: EMPLOYED'])

#### Numerical variables indication
The user has to indicate which are the numerical variables from the dataset.
	
	seg_data_num = ['ORIGINAL_BOOKED_AMOUNT','OUTSTANDING','BIRTH_DATE','DOWN_PAYMENT','REPORTING_DATE_DAYS_LAPSED','LOAN_OPEN_DATE_DAYS_LAPSED','EXPECTED_CLOSE_DATE_DAYS_LAPSED','CUSTOMER_OPEN_DATE_DAYS_LAPSED']

#### Run One Hot Encoder

One Hot Encoder will be use to transform the categorical variables.

	myrdf.encod(seg_data_cat)

#### Run the model for numerical values

	myrdf.set_train_num(seg_data_cat,target_value,seg_data_num)

The output of this method will indicate the GINI score for the full model and the segmented model, for numerical variables, and will be indicated below whether the split model is better or worse than the full model.

## Please cite the package in publications!
By using `Jigglypuff` you agree to the following rules:

You must place the following URL in a footnote to help others find `Jigglypuff`: https://github.com/hana-am/GroupF_RBASegmentation#data-handling

You assume all risk for the use of `Jigglypuff`.

Alexander, J., Almashari, H., Galeano, A., Godinez, F., Hernández, T. (2021a).



