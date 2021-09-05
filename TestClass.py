import RiskDataframe as rdf
import pandas as pd

dataframe = pd.read_csv("AUTO_LOANS_DATA.csv", sep=";")

pivot_value = 'ACCOUNT_NUMBER'
target_value = 'BUCKET'
down_payment = 'PROGRAM_NAME'
income_status = 'PROFESSION'
birth_date = 'BIRTH_DATE'
dates_todays = ['REPORTING_DATE','LOAN_OPEN_DATE','EXPECTED_CLOSE_DATE','CUSTOMER_OPEN_DATE']


myrdf = rdf.RiskDataframe(dataframe)
df = myrdf.start(pivot_value,birth_date,target_value, down_payment,income_status,dates_todays)

