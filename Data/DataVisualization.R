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



##### VAR MODEL -----
var.auto <- ar(data[1:1077,2:11], order.max = 20)
var.auto$aic #best one is p=2

qqnorm(var.auto$resid)
qqline(var.auto$resid,col=2)

varModel = VAR(data[1:1076,2:11], p=3)
vars.test <- function(x){
  norm <- sapply(normality.test(x)[[2]],function(xx) xx$p.value)
  arch <- arch.test(x)[[2]]$p.value
  ser <- serial.test(x)[[2]]$p.value
  return(c(norm,"arch"=arch,"serial"=ser))
}


vars.test(varModel) #testing normality
#not met at all with p=0 for all tests

#Seeing if const, trend, both or none is the best option
VARselect(data[,2:11], type="const", lag.max = 20) 
VARselect(data[,2:11], type="trend", lag.max = 20) 
VARselect(data[,2:11], type="both", lag.max = 20) 
VARselect(data[,2:11], type="none", lag.max = 20) 
#Best one is const

#Using type = const
(varModel1 <- VARselect(data[1:1077,2:11], type = "const", lag.max = 20))
#can model based on AIC, HQ, SC or FPE. AIC suggest order 2, SC suggest 1
(varModel = VAR(data[1:1077,2:11], p=3))
summary(varModel)
residualMatrix1 = as.matrix(residuals(varModel))
# a lot of insignificant variables - time to restrict the model
varModelrestrict = restrict(varModel, method = "ser")
summary(varModelrestrict) #only significant now

residualMatrix = as.matrix(residuals(varModelrestrict))
par(mfrow=c(3,2))
Acf(residualMatrix1[,1], lag.max = 60, main = "ACF of Consumer Returns")
Acf(residualMatrix1[,2], lag.max = 60, main = "ACF of Manufacturing Returns")
Acf(residualMatrix1[,3], lag.max = 60, main = "ACF of High Tech Returns")
Acf(residualMatrix1[,4], lag.max = 60, main = "ACF of Healthcare Returns")
Acf(residualMatrix1[,5], lag.max = 60, main = "ACF of Other Returns")

par(mfrow=c(3,2))
Pacf(residualMatrix1[,1], lag.max = 60, main = "PACF of Consumer Returns")
Pacf(residualMatrix1[,2], lag.max = 60, main = "PACF of Manufacturing Returns")
Pacf(residualMatrix1[,3], lag.max = 60, main = "PACF of High Tech Returns")
Pacf(residualMatrix1[,4], lag.max = 60, main = "PACF of Healthcare Returns")
Pacf(residualMatrix1[,5], lag.max = 60, main = "PACF of Other Returns")

ks.test(var.auto$resid[,1]/sd(var.auto$resid[,1],na.rm = TRUE),"pnorm")
ks.test(var.auto$resid[,2]/sd(var.auto$resid[,2],na.rm = TRUE),"pnorm")
## Cannot reject independence nor normality


(predObject = predict(varModelrestrict, n.ahead = 4))
predObject$fcst
predictions

varNoDurError = mean(abs(predObject$fcst$NoDur[,1] -data$NoDur[1078:1081]))
varDurblError = mean(abs(predObject$fcst$Durbl[,1]-data$Durbl[1078:1081]))
varManufError = mean(abs(predObject$fcst$Manuf[,1]-data$Manuf[1078:1081]))

(abs(predObject$fcst$NoDur[,1] -data$NoDur[1078:1081]))
/data$NoDur[1078:1081]


varx(data[800:1000, 2:11], PredDataOutputVAR)

par(mfrow=c(1,1))
plot(data$Manuf[1:1077], type = "l", 
     ylab= "Return", 
     xlab = "Observations",
     xlim=c(1060,1083), ylim=c(-20,30), main = "Forecasts for Manufacturing Return")
points(c(1078,1079,1080,1081),
       c(data$Manuf[1078],data$Manuf[1079],data$Manuf[1080],
         data$Manuf[1081]),
       col='black',pch=15)
points(c(1078,1079,1080,1081),
       c(predObject$fcst$Manuf[,1]),
       col='red',pch=16)
points(c(1078,1079,1080,1081),
       c(predObject$fcst$Manuf[,2]),
       col='red',pch=4)
points(c(1078,1079,1080,1081),
       c(predObject$fcst$Manuf[,3]),
       col='red',pch=4)
segments(1078,predObject$fcst$Manuf[1,3],1078,predObject$fcst$Manuf[1,2],col= 'red', lty = 2)
segments(1079,predObject$fcst$Manuf[2,3],1079,predObject$fcst$Manuf[2,2],col= 'red', lty = 2)
segments(1080,predObject$fcst$Manuf[3,3],1080,predObject$fcst$Manuf[3,2],col= 'red', lty = 2)
segments(1081,predObject$fcst$Manuf[4,3],1081,predObject$fcst$Manuf[4,2],col= 'red', lty = 2)
legend("topleft", c("Observed Values","Predicted Values", 
                    "Prediction Interval Bounds"), pch= c(16,16,4),
       col = c("black","red","red"))

##### VARX MODEL -----
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

originalData = data[,2:11]
dataCols = ncol(originalData)
#Removing NaNs
for(i in 1:dataCols){
  list = which(originalData[,i] == "NaN")
  originalData[list,i] = 0
  list = NULL
}

outputRows = nrow(PredDataOutput)
outputX = originalData[1:(outputRows),]
library(MTS)
VARX(outputX[900:1000,], 3, PredDataOutput[900:1000,3], m=1)

VARXFit(outputX,3,"AIC",list(k=1,s=4))



##### OUTPUT FILE WITH VIX-----
#Import vix
VixData = read.csv("VIXCLSMonthly.csv", header = F, sep = ";")
names(VixData)[1] = "YMD"
VixData = VixData[2:339,]
VixData$Month = substr(VixData$YMD, 6,7)
VixData$Year = substr(VixData$YMD, 1, 4)
VixData$YM = paste(VixData$Year, VixData$Month, sep="")

MatchList = match(data$YM, VixData$YM)
startRowVIX = which(MatchList %in% 1)

data = data[startRowVIX:nrow(data),]

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
outputY = originalData[2:(outputRows), 1]
outputX = originalData[1:(outputRows-1),]

VIXOutput= VixData$V2[1:outputRows-1]
outputData = cbind(outputX, PredDataOutput, VIXOutput, outputY)
write.table(outputData, file = "monthlyNoDurReturnVIX.csv", col.names = FALSE, row.names = F, sep = ",")



##### LOG-OUTPUT FILE WITH VIX -----
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
data = read.csv("Monthly - Average Equal Weighted Returns LOGRETURN.csv", header =T, sep = ";")
names(data)[1] = "YM"

#Import predictors
PredData = read.csv("PredictorData2016.csv", header =T, sep = ";")
predCols = ncol(PredData)
predRows = nrow(PredData)
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
PredData$IndexDiff = 0
for(i in 2:predRows){
  PredData$IndexDiff[i] = PredData$Index[i]/PredData$Index[i-1]
}
PredData$D12Diff = 0
for(i in 2:predRows){
  PredData$D12Diff[i] = PredData$D12[i]/PredData$D12[i-1]
}

PredData$E12Diff = 0
for(i in 2:predRows){
  PredData$E12Diff[i] = PredData$E12[i]/PredData$E12[i-1]
}
predCols2 = ncol(PredData)

PredData = PredData[,c(1,21,22,23, 5:20)]

MatchList = match(PredData$YM, data$YM)
startRow = which(MatchList %in% 1)
PredDataOutput = PredData[startRow:nrow(PredData),1:predCols]
PredDataOutput = PredDataOutput[1:nrow(PredDataOutput)-1,]
outputRows = nrow(PredDataOutput)+1 #to account for the shift in index data

data$YM[1]
PredDataOutput$YM[1]

data = data[1:outputRows,]
#there should a shift of 1 between the two dates underneath
data$YM[nrow(data)]
PredDataOutput$YM[nrow(PredDataOutput)]

#INCORPORATING VIX
VixData = read.csv("VIXCLSMonthly.csv", header = F, sep = ";")
names(VixData)[1] = "YMD"
VixData = VixData[2:339,]
VixData$Month = substr(VixData$YMD, 6,7)
VixData$Year = substr(VixData$YMD, 1, 4)
VixData$YM = paste(VixData$Year, VixData$Month, sep="")

VIX = numeric(outputRows-1)

MatchList = match(data$YM, VixData$YM)
startRowVIX = which(MatchList %in% 1)
endingObservation = length(VIX[startRowVIX:(outputRows-1)])
VIX[startRowVIX:(outputRows-1)] = VixData$V2[1:endingObservation]

PredDataOutput = PredDataOutput[,2:predCols]

## Incorporating recession data
RecessionData = read.csv("NBERRecessions.csv", header = T, sep = ";")
names(RecessionData)[1] = "YM"

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
outputY = originalData[2:(outputRows), 1]
outputX = originalData[1:(outputRows-1),]

predictionTime = data$YM[2:outputRows] #dates are the date of the Y variable, so we predict january '12, but have december '11 info available
outputData = cbind(outputX, PredDataOutput, VIX, outputY, predictionTime, RecessionData$Recession)
write.table(outputData, file = "monthlyNoDurLOGReturn2.csv", col.names = FALSE, row.names = F, sep = ",")
