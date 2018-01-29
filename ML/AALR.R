# This is Skipper af Rosenborg's and Esben Jørgensen Bager's implementation of 
#"An Algorithmic Approach to Linear Regression" by Dimitris Bertsimas, Angela King

# Clear environment
rm(list=ls())

# Load Data, missing values are noted with -99.99 or -999 and replaced with "NA"
library(readr)
#Daily data from 1st July 1926 to 31st October 2017
DailyAverageEqualWeightedReturns  <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Daily - Average Equal Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))
DailyAverageValueWeightedReturns  <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Daily - Average Value Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))

#Monthly Data from July 1926 - October 2017
MonAverageEqualWeightedReturns    <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Monthly - Average Equal Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))
MonAverageValueWeightedReturns    <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Monthly - Average Value Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))
MonNumberOfFirmsInPortfolios.csv  <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Monthly - Number of Firms in Portfolios.csv", skip = 1, na = c("-99.99", "-999","NA"))
MonAverageFirmSize                <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Monthly - Average Firm Size.csv", skip = 1, na = c("-99.99", "-999","NA"))

#Yearly data from 1927-2016
AnnAverageEqualWeightedReturns     <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Annual - Average Equal Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))
AnnAverageValueWeightedReturns     <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Annual - Average Value Weighted Returns.csv", skip = 1, na = c("-99.99", "-999","NA"))

#Yearly data from 1926-2017
AnnSumOfBESumOfMERatio              <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Annual - Sum of BE : Sum of ME.csv", skip = 1, na = c("-99.99", "-999","NA"))
AnnValueWeightedAverageOfBEMERatio <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Annual - Value-Weighted Average of BE:ME.csv", skip = 1, na = c("-99.99", "-999","NA"))
