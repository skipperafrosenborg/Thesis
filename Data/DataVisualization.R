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


setwd("C:/Users/ejb/Documents/GitHub/Thesis/Data/MonthlyReturns")
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
  boxplot(data[,i] ~ data$Month, main = paste("Monthly Boxplots for", names(data)[i], "Returns"),ylim=c(-20, 20))
}
for(i in (split+2):initCols-1){
  boxplot(data[,i] ~ data$Month, main = paste("Monthly Boxplots for", names(data)[i], "Returns"),ylim=c(-20, 20))
}
#No statistical difference between months at all

#Years to see if yearly seasonality could be present
plot.new()
par(mfrow=c(plotRows, plotCols))
for(i in 2:split){
  boxplot(data[,i] ~ data$Year, main = paste("Yearly Boxplots for", names(data)[i], "Returns"),ylim=c(-20, 20))
}
for(i in (split+1):(initCols-1)){
  boxplot(data[,i] ~ data$Year, main = paste("Yearly Boxplots for", names(data)[i], "Returns"),ylim=c(-20, 20))
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


##### OUTPUT FILE -----

#Import predictors
PredData = read.csv("PredictorData2016.csv", header =T, sep = ";")
predCols = ncol(PredData)
names(PredData)[1] = "YM"
YM = PredData$YM
PredData$Month = substr(PredData$YM, 5,6)
PredData$Year = substr(PredData$YM, 1, 4)
#Removing NaNs
for(i in 1:predCols){
  list = which(PredData[,i] == "NaN")
  PredData[list,i] = 0
  list = NULL
}

MatchList = match(PredData$YM, data$YM)
startRow = which(MatchList %in% 1)
PredDataOutput = PredData[startRow:nrow(PredData),2:predCols]
PredDataOutput = PredDataOutput[1:nrow(PredDataOutput)-1,]
outputRows = nrow(PredDataOutput)+1 #to account for the shift in index data

#INDEX DATA FOR OUTPUT
originalData = data[,2:11]
dataCols = ncol(originalData)
#Removing NaNs
for(i in 1:dataCols){
  list = which(originalData[,i] == "NaN")
  originalData[list,i] = 0
  list = NULL
}

#Shift data
outputY = originalData[2:(outputRows), 9]
outputX = originalData[1:(outputRows-1),]


outputData = cbind(outputX, PredDataOutput, outputY)
write.table(outputData, file = "monthlyUtilsReturn.csv", col.names = FALSE, row.names = F, sep = ",")

##### CORRELATIONS -----
cor(combinedData[,7:combCols])
#CRSPvw = CRSP value weighted
#CRSPvwx = CRSP value weighted without dividends
