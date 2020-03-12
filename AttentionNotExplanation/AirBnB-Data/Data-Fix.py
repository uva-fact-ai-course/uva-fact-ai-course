import pandas as pd
import numpy as np
import re

#=============================#
#        Data Creation        #
#=============================#
#creation of dataframe
#select a city from one of the three down below and change the boolean to true if you wish a single query.
# The result is a pandas data frame with the index of ID and columns containing review scores.

def createData(city):

    df = pd.read_csv(city)
    names = list(df)
    
    #using regex to go through the list searching out all names containing review.
    r = re.compile("^review.*")
    names_check = list(filter(r.match,names))
    names_check.insert(0,'id')
    names_check.remove('reviews_per_month')   
    #new_columns = names_check
    new_columns = ['id', 'review_scores_rating']

    df2 = cleanup(df, names, names_check, new_columns)
    return df2

#cleanup step in the dataframe creation.
def cleanup(data, names, names_check, new_columns):
    #There is a difference between single_query and multi_query in data.
    #Thus we go over the twice in the case of single_query. To make sure they are equally long
    #First we go over the multi_query after which we drop all nans
    #If we wish a single query we THEN go over that list to drop the rest of the columns. 

    for x in names:
        if x not in names_check:
            del data[x]

    data = data.dropna()
    multi_query_data = data
    multi_query_data = multi_query_data.set_index('id')
    #Dropping to make it a single query
    if new_columns != names_check:
        for x in names_check:
            if x not in new_columns:
                del data[x]

    data = data.set_index('id')

    return data, multi_query_data

#=============================#
#          Ranking            #
#=============================#
# ?


#=============================#
#        Helper Stuff         #
#=============================#


#The listings are taken from insideairbnb.com, format: listings-city-D/M/Y. 
#Biege et al. Don't mention as to which year they used.
#They make no mention of how they cleaned the data. 
#So after removing all irrelevant tables we dropped all rows containing NaNs this ensures that the single query dataframe is equally long to the multi query dataframe.
#This function takes a string and a boolean to determine which multi_query and single_query should be returned
def clean_data(city="Hong Kong", current_date = False):

    cities = {"Boston": ["listings-boston-06-10-2017.csv", "listings-boston-04-12-2019.csv"],
              "Geneva": ["listings-geneva-17-11-2017.csv", "listings-geneva-28-11-2019.csv"],
              "Hong Kong": ["listings-hongkong-07-08-2016.csv", "listings-hongkong-19-11-2019.csv"]}

    if current_date:

        data, multi_query_data = createData(cities[city][1])
    else:
        data, multi_query_data = createData(cities[city][0])

    return data, multi_query_data


data1, data2 = clean_data()
#print(data1)
data2.to_csv('hongkong.csv')


