# -*- coding: utf-8 -*-
"""
Created on Fri Sep 3
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from copy import copy
from sklearn.preprocessing import OneHotEncoder




#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
class RiskDataframe(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes to a prticular
    type of Dataframes in the Risk Indistry. 
    It provides the end user with both general and specific cleaning functions, 
    though they never reference a specific VARIABLE NAME.
    
    It facilitates the End User to perform some Date Feature Engineering,
    Scaling, Encoding, etc. to avoid code repetition.
    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data = args[0]

    
    @property
    def _constructor(self):
        def func_(*args,**kwargs):
            df = RiskDataframe(*args,**kwargs)
            return df
        return func_
    
#-----------------------------------------------------------------------------
                        # DATA HANDLING
#-----------------------------------------------------------------------------

    def SetAttributes(self, kwargs):
        """
        The function will update the type of the variable submitted for change.
        It will veify first that the key is present in the desired dataframe.
        If present, it will try to change the type to the desired format.
        If not possible, it will continue to the next element.         
        Parameters
        ----------
        **kwargs : The key-argument pair of field-type relationship that
        wants to be updated.
        Returns
        -------
        None.
        """
        if self.shape[0] > 0:
            for key,vartype in kwargs.items():
                if key in self.columns:
                    try:
                        self[key] = self[key].astype(vartype)
                    except:
                        print("Undefined type {}".format(str(vartype)))
                else:
                    print("The dataframe does not contain variable {}.".format(str(key)))
        else:
            print("The dataframe has not yet been initialized")

#-----------------------------------------------------------------------------
                        # RISK BASED APPROACH
#-----------------------------------------------------------------------------

    def heatPlot(self):
        """
         The function will creat a heatmap graph for the correlated of the attributes

         Parameters
         ----------
         None

         Returns
         -------
         None.

         """
        mycor_1 = self.corr()
        plt.figure(figsize=(10, 10))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(mycor_1, xticklabels=mycor_1.columns.values,
                    yticklabels=mycor_1.columns.values, cmap=cmap, vmax=1, vmin=-1, center=0, square=True,
                    linewidths=.5, cbar_kws={"shrink": .82})

        plt.title('Heatmap of Correlation Matrix Personal')

    def missing_not_at_random(self, input_vars=[] ):
        """
        The function is for presenting the missing not at random analysis of the variables

        Parameters
        ----------
        input_vars : Dataframe variables

        Returns
        -------
        The print of the analysis .

        """
        for var in input_vars:
          if var not in self.columns:
            print(f"Variable named {var} not in the dataframe. Review the input variable names")
            return
        if input_vars==[]: columns = self.columns
        else: columns = input_vars
        missing_value_columns = [column for column in columns if self[column].isnull().values.any()]
        print(f"Missing Not At Random Repport (MNAR) - {', '.join(missing_value_columns) if len(missing_value_columns)>0 else 'No'} variables seem Missing Not at Random, there for we recommend: \n \n Thin File Segment Variables (all others variables free of MNAR issue): {', '.join([column for column in columns if column not in missing_value_columns])} \n \n Full File Segment Variables: {', '.join(columns)}")
        return

    def highly_correlated_variables(self, target, top_most=4):
        """
        The function is for presenting the correlated variables to the target

        Parameters
        ----------
        Dataframe variables
        target : The target variable in the data frame
        Returns
        -------
        The print of the analysis .

        """
        missing_value_columns = [column for column in self.columns if self[column].isnull().values.any()]
        correlated_columns = list(self.corr().sort_values(by = target)[target].index)
        print(f"\n The highly correlated columns are {', '.join([column for column in correlated_columns if column not in missing_value_columns][:top_most])}")
        return


    # -----------------------------------------------------------------------------
    # DATA CLEANING
    # -----------------------------------------------------------------------------

    def start(self, piv, birth_date, target, down_payment, income_status, dates_todays):
        """
        The function is for start preprocessing the variables, and it call other methods/functions

        Parameters
        ----------
        piv:pivot value
        birth_date: birth date
        target:Target value
        down_payment:down payment value
        income_status: income status value
        Dates_todays:a list of all the dates

        Returns
        -------
        The cleaned and process dataset .

        """
        self._pivot_unique(piv)
        self._clean_target(target)
        self._clean(birth_date)
        self._down(down_payment)
        self._income(income_status)
        self._dayslapsed(dates_todays)
        return self.data

    def _pivot_unique(self, piv):
        """
         The function is to remove  duplicates using the 'pivot' value established by the user

         Parameters
         ----------
         piv:pivot value

         Returns
         -------
         The dataframe.

         """
        self.data.drop_duplicates(subset=[piv], keep='last', inplace=True)
        return self.data


    def _clean_target(self, target):
        """
         The function is to to have a binary Target

         Parameters
         ----------
         target:target value

         Returns
         -------
         The dataframe.

        """
        for i in range(len(self.data.columns)):
            tar = str(self.data.columns[i])
            if tar == target:
                val = np.where(self.data[self.data.columns[i]] > 0, 1, self.data[self.data.columns[i]])
                self.data[self.data.columns[i]] = val
        return self.data


    def _clean(self, birth_date):
        """
         The function is for Getting the age from the birth date in the dataFrame and cleanig empty spaces
            - clean numerical values
            -  clean categorical values
            -  fill empty

         Parameters
         ----------
         birth_date :birth date  value

         Returns
         -------
         The dataframe.

        """
        data = Dataset.from_dataframe(self.data)
        numerical_features = data.numerical_features
        categorical_features = data.categorical_features


        for i in range(len(self.data.columns)):
            empty = self.data[self.data.columns[i]].isna().any()
            if empty == True and self.data.columns[i] in numerical_features:
                val = self.data[self.data.columns[i]].fillna(self.data[self.data.columns[i]].mean())
                self.data[self.data.columns[i]] = val


        for i in range(len(self.data.columns)):
            empty = self.data[self.data.columns[i]].isna().any()
            if empty == True and self.data.columns[i] in categorical_features:
                val = self.data[self.data.columns[i]].fillna('UNKNOWN')
                self.data[self.data.columns[i]] = val

            # upper case
        for i in range(len(self.data.columns)):
            if self.data.columns[i] in categorical_features:
                val = self.data[self.data.columns[i]].str.upper()
                self.data[self.data.columns[i]] = val

        # if birth date, return age
        def age(born):
            while True:
                try:
                    born = datetime.strptime(born, "%Y-%m-%d").date()
                    today = date.today()
                    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
                except ValueError:
                    return 'UNKNOWN'

        for i in range(len(self.data.columns)):
            dateb = str(self.data.columns[i])
            if self.data.columns[i] in categorical_features and dateb == birth_date:
                val = self.data[self.data.columns[i]].apply(age)
                self.data[self.data.columns[i]] = val

        return self.data

    def _down(self, down_payment):
        """
        The function to separate between individuals and corporate as the down Payment has to mean something in order to be useful in the Model. We get the % in each value.

         Parameters
         ----------
         down_payment :down payment value

         Returns
         -------
         The dataframe.

        """
        def pay(payment):
            y = re.findall('\d+', payment)
            if len(y) > 0:
                result = int(y[0]) / 100
            else:
                result = 0
            return result

        def ty(types):
            if 'EMPLOYED' in types:
                return 'EMPLOYED'
            else:
                return 'CORPORATE'

        self.data["DOWN_PAYMENT"] = None
        self.data["TYPE"] = None
        val = self.data[down_payment].apply(pay)
        self.data["DOWN_PAYMENT"] = val

        val = self.data[down_payment].apply(ty)
        self.data["TYPE"] = val
        self.data.drop(columns=[down_payment], inplace=True)

        return self.data

    def _income(self, income_status):
        """
        There are too many jobs,so this function isolates between those who get some income and unemployed guys

         Parameters
         ----------
         income_status :income status value

         Returns
         -------
         The dataframe.

        """
        def income(incomes):
            if 'UNEMPLOYED' in incomes:
                return 'UNEMPLOYED'
            else:
                return 'ACTIVE'

        val = self.data[income_status].apply(income)
        self.data[income_status] = val


    def _dayslapsed(self, dates_todays):
        """
     The function is transferring the Dates to numerical values.

         Parameters
         ----------
         dates_todays :a list of dates variables

         Returns
         -------
         The dataframe.

        """
        def daily(dai):
            change = datetime.strptime(dai, "%Y-%m-%d").date()
            today = date.today()
            delta = today - change
            return delta.days

        for i in range(len(dates_todays)):
            self.data[dates_todays[i] + '_DAYS_LAPSED'] = None
            val = self.data[dates_todays[i]].apply(daily)
            self.data[dates_todays[i] + '_DAYS_LAPSED'] = val
            self.data.drop(columns=[dates_todays[i]], inplace=True)

    # -----------------------------------------------------------------------------
    # DATA ANALYSIS
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # CATEGORICAL VARIABLES ANALYSIS
    # -----------------------------------------------------------------------------

    def set_train_cat(self, target_value, seg_data):
        """
        The function is segmenting the categorical variables

         Parameters
         ----------
         target_value :target value
         seg_data : categorical variables in the dataframe

         Returns
         -------
         The dataframe.

        """
        df_random_sample, _ = train_test_split(self.data, test_size=0.90)

        def get_specific_columns(df_random_sample, data_types, to_ignore=list(), ignore_target=False):
            columns = df_random_sample.select_dtypes(include=data_types).columns
            if ignore_target:
                columns = filter(lambda x: x not in to_ignore, list(columns))
            return list(columns)

        all_numeric_variables = get_specific_columns(df_random_sample, ["float64", "int64"], [target_value],
                                                     ignore_target=True)

        splitter = train_test_split
        df_train, df_test = splitter(df_random_sample, test_size=0.2, random_state=42)

        X_train = df_train[all_numeric_variables]
        y_train = df_train[target_value]

        X_test = df_test[all_numeric_variables]
        y_test = df_test[target_value]

        method = LogisticRegression(random_state=0)
        fitted_full_model = method.fit(X_train, y_train)
        y_pred = fitted_full_model.predict(X_test)

        # Result accuracy all model
        full_model = accuracy_score(y_test, y_pred)
        result_full_model_etal = [
            "The total accuracy using all variable and Logistic regression is: " + str(full_model)]

        conclusion_model = []

        for seg in range(len(seg_data)):

            max_value_seg = self.data[seg_data[seg]].value_counts().idxmax()

            # set dataframes of train and test

            df_train_seg1 = df_train[df_random_sample[seg_data[seg]] == max_value_seg]
            df_train_seg2 = df_train[df_random_sample[seg_data[seg]] != max_value_seg]
            df_test_seg1 = df_test[df_random_sample[seg_data[seg]] == max_value_seg]
            df_test_seg2 = df_test[df_random_sample[seg_data[seg]] != max_value_seg]

            # getting results seg 1 vs seg 1

            X_train_seg1 = df_train_seg1[all_numeric_variables]
            y_train_seg1 = df_train_seg1[target_value]
            X_test_seg1 = df_test_seg1[all_numeric_variables]
            y_test_seg1 = df_test_seg1[target_value]

            fitted_model_seg1 = method.fit(X_train_seg1, y_train_seg1)

            def GINI(y_test, y_pred_probadbility):
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_probadbility)
                roc_auc = auc(fpr, tpr)
                GINI = (2 * roc_auc) - 1
                return (GINI)

            y_pred_seg1_proba = fitted_model_seg1.predict_proba(X_test_seg1)[:, 1]
            y_pred_seg1_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg1)[:, 1]

            result_full_model_etal.append("Using: " + seg_data[seg] + " GINI Full Model Seg1: " + str(
                GINI(y_test_seg1, y_pred_seg1_proba) * 100) + "%")
            result_full_model_etal.append("Using: " + seg_data[seg] + " GINI Segmented Model Seg1: " + str(
                GINI(y_test_seg1, y_pred_seg1_fullmodel_proba) * 100) + "%")

            # getting results seg 2 vs seg 2

            X_train_seg2 = df_train_seg2[all_numeric_variables]
            y_train_seg2 = df_train_seg2[target_value]
            X_test_seg2 = df_test_seg2[all_numeric_variables]
            y_test_seg2 = df_test_seg2[target_value]
            fitted_model_seg2 = method.fit(X_train_seg2, y_train_seg2)
            y_pred_seg2 = fitted_model_seg2.predict(X_test_seg2)
            y_pred_seg2_fullmodel = fitted_full_model.predict(X_test_seg2)

            y_pred_seg2_proba = fitted_model_seg1.predict_proba(X_test_seg2)[:, 1]
            y_pred_seg2_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg2)[:, 1]

            result_full_model_etal.append("Using: " + seg_data[seg] + " GINI Full Model Seg2: " + str(
                GINI(y_test_seg2, y_pred_seg2_proba) * 100) + "%")
            result_full_model_etal.append("Using: " + seg_data[seg] + " GINI Segmented Model Seg2:" + str(
                GINI(y_test_seg2, y_pred_seg2_fullmodel_proba) * 100) + "%")

            if GINI(y_test_seg1, y_pred_seg1_proba) * 100 < 20 or GINI(y_test_seg2, y_pred_seg2_proba) * 100 < 20:

                conclusion_model.append("After analysis, we did not find a good split using: " + seg_data[seg])
            else:
                conclusion_model.append(
                    "After analysis, we find a good split using: " + seg_data[seg] + " set at: " + str(max_value_seg))

        return result_full_model_etal, conclusion_model

# -----------------------------------------------------------------------------
# ENCODING
# -----------------------------------------------------------------------------

    def encod(self, seg_data_cat):
        """
        The function is encoding the categorical variables

         Parameters
         ----------
         seg_data_cat : categorical variables in the dataframe

         Returns
         -------
         The dataframe.

        """
        data = Dataset.from_dataframe(self.data)
        for seg in range(len(seg_data_cat)):
            data.onehot_encode(seg_data_cat[seg])
            data.drop_columns(seg_data_cat[seg])

        self.data = data.features

        return self.data

# -----------------------------------------------------------------------------
# NUMERICAL VARIABLES ANALYSIS
# -----------------------------------------------------------------------------

    def set_train_num(self, seg_data_cat, target_value, seg_data_num):
        """
        The function is segmenting the numerical variables

         Parameters
         ----------
         target_value :target value
         seg_data_num : numerical variables in the dataframe

         Returns
         -------
         The dataframe.

        """
        # Lets get rid of Unknown values so that we can have means in each column
        for dro in range(len(seg_data_num)):
            self.data.drop(self.data.index[self.data[seg_data_num[dro]] == 'UNKNOWN'], inplace=True)

        def get_specific_columns(df_random_sample, data_types, to_ignore=list(), ignore_target=False):
            columns = df_random_sample.select_dtypes(include=data_types).columns
            if ignore_target:
                columns = filter(lambda x: x not in to_ignore, list(columns))
            return list(columns)

        df_random_sample, _ = train_test_split(self.data, test_size=0.90)
        all_numeric_variables = get_specific_columns(df_random_sample, ["float64", "int64"], [target_value],
                                                     ignore_target=True)

        splitter = train_test_split
        df_train, df_test = splitter(df_random_sample, test_size=0.2, random_state=42)

        result_full_model_etal = []

        X_train = df_train[all_numeric_variables]
        y_train = df_train[target_value]

        X_test = df_test[all_numeric_variables]
        y_test = df_test[target_value]

        method = LogisticRegression(random_state=0)
        fitted_full_model = method.fit(X_train, y_train)
        y_pred = fitted_full_model.predict(X_test)

        full_model = accuracy_score(y_test, y_pred)
        result_full_model_etal = [
            "The total accuracy using all variable and Logistic regression is: " + str(full_model)]

        conclusion_model = []

        for seg in range(len(seg_data_num)):

            mean_value_seg = self.data[seg_data_num[seg]].mean()

            # set dataframes of train and test

            df_train_seg1 = df_train[df_random_sample[seg_data_num[seg]] >= mean_value_seg]
            df_train_seg2 = df_train[df_random_sample[seg_data_num[seg]] < mean_value_seg]
            df_test_seg1 = df_test[df_random_sample[seg_data_num[seg]] >= mean_value_seg]
            df_test_seg2 = df_test[df_random_sample[seg_data_num[seg]] < mean_value_seg]

            # getting results seg 1 vs seg 1

            X_train_seg1 = df_train_seg1[all_numeric_variables]
            y_train_seg1 = df_train_seg1[target_value]
            X_test_seg1 = df_test_seg1[all_numeric_variables]
            y_test_seg1 = df_test_seg1[target_value]

            fitted_model_seg1 = method.fit(X_train_seg1, y_train_seg1)

            def GINI(y_test, y_pred_probadbility):
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_probadbility)
                roc_auc = auc(fpr, tpr)
                GINI = (2 * roc_auc) - 1
                return (GINI)

            y_pred_seg1_proba = fitted_model_seg1.predict_proba(X_test_seg1)[:, 1]
            y_pred_seg1_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg1)[:, 1]

            result_full_model_etal.append("Using: " + seg_data_num[seg] + " GINI Full Model Seg1: " + str(
                GINI(y_test_seg1, y_pred_seg1_proba) * 100) + "%")
            result_full_model_etal.append("Using: " + seg_data_num[seg] + " GINI Segmented Model Seg1: " + str(
                GINI(y_test_seg1, y_pred_seg1_fullmodel_proba) * 100) + "%")

            # getting results seg 2 vs seg 2

            X_train_seg2 = df_train_seg2[all_numeric_variables]
            y_train_seg2 = df_train_seg2[target_value]
            X_test_seg2 = df_test_seg2[all_numeric_variables]
            y_test_seg2 = df_test_seg2[target_value]
            fitted_model_seg2 = method.fit(X_train_seg2, y_train_seg2)
            y_pred_seg2 = fitted_model_seg2.predict(X_test_seg2)
            y_pred_seg2_fullmodel = fitted_full_model.predict(X_test_seg2)

            y_pred_seg2_proba = fitted_model_seg1.predict_proba(X_test_seg2)[:, 1]
            y_pred_seg2_fullmodel_proba = fitted_full_model.predict_proba(X_test_seg2)[:, 1]

            result_full_model_etal.append("Using: " + seg_data_num[seg] + " GINI Full Model Seg2: " + str(
                GINI(y_test_seg2, y_pred_seg2_proba) * 100) + "%")
            result_full_model_etal.append("Using: " + seg_data_num[seg] + " GINI Segmented Model Seg2: " + str(
                GINI(y_test_seg2, y_pred_seg2_fullmodel_proba) * 100) + "%")

            if GINI(y_test_seg1, y_pred_seg1_proba) * 100 < 20 or GINI(y_test_seg2, y_pred_seg2_proba) * 100 < 20:
                conclusion_model.append("After analysis, we did not find a good split using: " + seg_data_num[seg])
            else:
                conclusion_model.append(
                    "After analysis, we find a good split using: " + seg_data_num[seg] + " set at: " + str(
                        mean_value_seg))

        return result_full_model_etal, conclusion_model

    def plot_risk (self,variable):
        """
        The function is for ploting the dataframe variables

         Parameters
         ----------
         variable :any variable in the dataframe

         Returns
         -------
         The creating plots.

        """
        plt.hist(self.data[variable], color='g', label='Ideal')
        print(self.data.describe())