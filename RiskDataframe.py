# -*- coding: utf-8 -*-
"""
Created on Fri Sep 3
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import _tree
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

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
    def missing_not_at_random(self, input_vars=[] ):
        """
        Returns
        -------
        A print with the analysis.

        We can plot the anlyais of the heatmap
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

# -----------------------------------------------------------------------------
    def _datetime_to_seconds(self, string_value):

            """
            Returns
            -------
            The functions to convert datetime into seconds
            """

            try:
                obj = datetime.strptime(string_value, '%Y-%m-%d %H:%M:%S')
            except:
                obj = datetime.strptime(string_value, '%Y-%m-%d')
            time_delta = datetime(1970, 1, 1)
            seconds = int((obj - time_delta).total_seconds())

            return seconds

    def _seconds_to_datetime(self, seconds):
        """
        Returns
        -------
        The functions to convert seconds into datetime
        """
        return (datetime.fromtimestamp(seconds)).strftime('%Y-%m-%d %H:%M:%S')


    def _handle_missing_values(self):
        """
        The functions handle the missing values with two cases
        fill it with the mode if it's not Numeric ( Float )
        fill it with the mean if it's with a type float
        """
        missing_value_columns = [column for column in self.columns if self[column].isnull().values.any()]
        for column in missing_value_columns:
            if self.dtypes[column] != np.dtype('float64'):
                self[column].fillna(self[column].mode()[0], inplace=True)
            else:
                self[column].fillna(self[column].mean(), inplace=True)

    def _handle_datetime_values(self):
        """
        convert the date time to seconds
        """

        datetime_columns = [column for column in self.columns if self.dtypes[column] == np.dtype('<M8[ns]')]
        for column in datetime_columns:
            self[column] = self[column].apply(lambda x: self.datetime_to_seconds(self,str(x)))

    def _handle_categorical_values(self):

        """
        Handling the categorical values by applying the labelEncoder

        """

        categorical_columns = [column for column in self.columns if
                               self.dtypes[column] not in [np.dtype('<M8[ns]'), np.dtype('float64'), np.dtype('int64')]]
        ## Initializing dictionary to store all the encoders as values and their respective columns as keys
        encode_decode = {}
        ## Running a for loop which would create create encoder for each categorical column and store it in initialized dictionary
        for column in categorical_columns:
            # Initialize encoder
            encoder = LabelEncoder()
            # Train encoder
            encoder.fit(myrdf[column])
            ## Using encoder to encode categorical columns in dataset
            self[column] = encoder.transform(self[column])
            ## Store encoder in dictionary
            encode_decode[column] = encoder
        self.encoders = encode_decode

    def clean_dataframe (self,string_value,seconds ):
        self._datetime_to_seconds(string_value)
        self._seconds_to_datetime(seconds)
        self._handle_missing_values()
        self._handle_datetime_values()
        self._handle_categorical_values()
        return self.data

