## Setting packages
library(MASS)
library(stats)
library(tseries)
library(forecast)
library(lubridate)
library(date)
library(zoo)
library(marima)
library(vars)
library(LSTS)

setwd("C:/Users/ejb/Documents/GitHub/Thesis/Data")
data = read.csv("Monthly - Average Equal Weighted Returns.csv", header =T)
data$Average = rowMeans(data[,2:11], na.rm = TRUE)
data$CumSumAverage = cumsum(data$Average)
initCols = ncol(data)
names(data)[1] = "YM"
YM = data$YM
data$Month = substr(data$YM, 5,6)
data$Year = substr(data$YM, 1, 4)
str(data)

## Number of observations for estimation and prediction
len = length(YM)

# Save dates
data$date = rep(NA, len)
data$date = as.yearmon(as.character(data$YM), "%Y%m")
str(data)

str(data)
nCols = ncol(data)
ceiling(10/3)

names(data)[2]
plotRows = 3

##### TIME-SERIES PLOTS -----

(plotCols = ceiling((initCols-1)/plotRows)/2)
par(mfrow=c(plotRows, plotCols))
split = ceiling(initCols/2)
for(i in 2:split){
  plot(data$date, data[,i], type = "l", xlab = 'Time', ylab = names(data)[i])
}

for(i in split:initCols){
  plot(data$date, data[,i], type = "l", xlab = 'Time', ylab = names(data)[i])
}


## Testing stationarity through Augmented Dickey-Fuller tests. The null-hypothesis is non-stationarity
#so for a low p-value, it is unlikely that it holds.
for(i in 2:initCols){
  print(paste("P-value for adf.test",i,"is: ", adf.test(data[,i])$p.value))
}
#all time-series are stationary, despite the peaks. Maybe a differencing anyway


##### BOXPLOTS -----

#Months to see if seasonality could be present
plot.new()
par(mfrow=c(plotRows, plotCols))
for(i in 2:split){
  boxplot(data[,i] ~ data$Month, main = paste("Monthly Boxplots for", names(data)[i], "Returns"),ylim=c(-30, 70))
}
for(i in (split+2):initCols-1){
  boxplot(data[,i] ~ data$Month, main = paste("Monthly Boxplots for", names(data)[i], "Returns"),ylim=c(-30, 70))
}
#No statistical difference between months at all

#Years to see if yearly seasonality could be present
plot.new()
par(mfrow=c(plotRows, plotCols))
for(i in 2:split){
  boxplot(data[,i] ~ data$Year, main = paste("Yearly Boxplots for", names(data)[i], "Returns"),ylim=c(-30, 70))
}
for(i in (split+1):(initCols-1)){
  boxplot(data[,i] ~ data$Year, main = paste("Yearly Boxplots for", names(data)[i], "Returns"),ylim=c(-30, 70))
}
#not exactly stationary, especially around financial crisis in 1929


##### HISTOGRAMS -----
#Histogram to see distribution of returns
plot.new()
par(mfrow=c(plotRows, plotCols))
for(i in 2:split){
  hist(data[,i], breaks = 250, main = paste("Distribution of ", names(data)[i], "Returns"))
}

for(i in (split+1):(initCols-1)){
  hist(data[,i], breaks = 250, main = paste("Distribution of ", names(data)[i], "Returns"))
}


##### EQUITY PREDICTION FACTORS -----
library(readxl)
predictorData = read.csv("PredictorData2016.csv", sep = ";", header = T)
predRows = dim(predictorData)[1]
names(predictorData)[1] = "YM2"
YM2 = predictorData$YM2


(startRange = match(data$YM[1], predictorData$YM))
(endRange   = match(predictorData$YM[predRows-1], data$YM))

combinedData = cbind(data[1:endRange, ], predictorData[startRange:(predRows-1),])
combinedData$YM2 = NULL
combCols = ncol(combinedData)
combinedData = combinedData[c(1, 12, 13, 14, 15, 16, c(2:11), c(17:combCols))]

#Removing NaNs
for(i in 1:combCols){
  list = which(combinedData[,i] == "NaN")
  combinedData[list,i] = 0
  list = NULL
}

write.csv(combinedData, file = "combinedIndexData.csv")

##### CORRELATIONS -----
cor(combinedData[,7:combCols])
#CRSPvw = CRSP value weighted
#CRSPvwx = CRSP value weighted without dividends
