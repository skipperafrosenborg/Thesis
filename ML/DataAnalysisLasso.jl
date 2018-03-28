using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Results/IndexData/LassoTests/12-1 VIX/"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"

stringArr = Array{String}(10)
for j = 1:10
    stringArr[j] = "gamma =$j"
end

function logStuff(raw, time, exp, TA)
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Results/IndexData/LassoTests/12-1 VIX/"
    tempString = ""
    if raw == 0
        tempString = tempString*"Macro"
    else
        tempString = tempString*"Raw"
    end

    if time == 1
        tempString = tempString*"Time"
    end

    if exp == 1
        tempString = tempString*"Exp"
    end

    if TA == 1
        tempString = tempString*"TA"
    end

    cd(path)
    y_real = CSV.read("121_Shrink_Raw_Real.CSV",
        delim = ',', nullable=false, header=["y_real"])

    y_hat = CSV.read("121_Shrink_"*tempString*"_predicted.CSV",
        delim = ',', nullable=false, header=stringArr, types=[Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64])

    y_real = y_real[1:end,:]
    y_hat = y_hat[1:end,:]

    nRows = size(y_hat)[1]
    nCols = size(y_hat)[2]

    classificationRate = Array{Float64}(nCols)
    meanErr = Array{Float64}(nCols)
    RMSE = Array{Float64}(nCols)
    Rsquare = Array{Float64}(nCols)
    SSTO = sum((y_real[i,1]-mean(y_real[:,1]))^2 for i=1:nRows)

    for g = 1:10
        currentCol = g
        # Calculate classifaction rate
        trueClass = 0
        for row = 1:nRows
            if sign(y_real[row,1]) == sign(y_hat[row,currentCol])
                trueClass += 1
            end
        end
        classificationRate[currentCol] = trueClass/nRows

        # Mean Error rate
        # Calculate RMSE
        # Calculate R^2
        errSum = 0
        errSquaredSum = 0
        for row = 1:nRows
            errSum += abs(y_real[row,1] - y_hat[row,currentCol])
            errSquaredSum += (y_real[row,1] - y_hat[row,currentCol])^2
        end
        meanErr[currentCol] = errSum/nRows
        RMSE[currentCol] = sqrt(errSquaredSum/nRows)
        Rsquare[currentCol] = 1- errSquaredSum/SSTO

        # Calculate insample error

        # Process data
    end

    tempArr = zeros(1,8)

    classRate = findmax(classificationRate)[1]
    classIndex = findmax(classificationRate)[2]
    OOSR = findmax(Rsquare)[1]
    OOSRIndex = findmax(Rsquare)[2]
    meanErrVal = findmax(meanErr)[1]
    meanErrIndex = findmax(meanErr)[2]
    RMSEVal = findmax(RMSE)[1]
    RMSEIndex = findmax(RMSE)[2]
    tempArr[1,1] = findmax(classificationRate)[1]
    tempArr[1,2] = findmax(Rsquare)[1]
    tempArr[1,3] = findmax(meanErr)[1]
    tempArr[1,4] = findmax(RMSE)[1]
    tempArr[1,5] = findmax(classificationRate)[2]
    tempArr[1,6] = findmax(Rsquare)[2]
    tempArr[1,7] = findmax(meanErr)[2]
    tempArr[1,8] = findmax(RMSE)[2]

    #println("Classification: ",findmax(classificationRate))
    #println("OoS R²: ",findmax(Rsquare))
    #println("Mean Absolute Error: ",findmin(meanErr))
    #println("RMSE: ",findmin(RMSE))

    return tempArr
end

counterIter = 1
logArr = zeros(16,8)

for raw = 0:1
	for time = 0:1
		for exp = 0:1
			for TA = 0:1
				logArr[counterIter,:] = logStuff(raw, time, exp, TA)
                counterIter += 1
			end
		end
	end
end

#=
raw = 0
time = 1
exp = 0
TA = 1
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Results/IndexData/LassoTests/240-1 VIX/"
tempString = ""
if raw == 0
    tempString = tempString*"Macro"
else
    tempString = tempString*"Raw"
end

if time == 1
    tempString = tempString*"Time"
end

if exp == 1
    tempString = tempString*"Exp"
end

if TA == 1
    tempString = tempString*"TA"
end

cd(path)
y_real = CSV.read("2401_Shrink_Raw_Real.CSV",
    delim = ',', nullable=false, header=["y_real"])

y_hat = CSV.read("2401_Shrink_"*tempString*"_predicted.CSV",
    delim = ',', nullable=false, header=stringArr, types=[Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64])

y_real = y_real[1:end,:]
y_hat = y_hat[1:end,:]

nRows = size(y_hat)[1]
nCols = size(y_hat)[2]

classificationRate = Array{Float64}(nCols)
meanErr = Array{Float64}(nCols)
RMSE = Array{Float64}(nCols)
Rsquare = Array{Float64}(nCols)
SSTO = sum((y_real[i,1]-mean(y_real[:,1]))^2 for i=1:nRows)

for g = 1:10
    currentCol = g
    # Calculate classifaction rate
    trueClass = 0
    for row = 1:nRows
        if sign(y_real[row,1]) == sign(y_hat[row,currentCol])
            trueClass += 1
        end
    end
    classificationRate[currentCol] = trueClass/nRows

    # Mean Error rate
    # Calculate RMSE
    # Calculate R^2
    errSum = 0
    errSquaredSum = 0
    for row = 1:nRows
        errSum += abs(y_real[row,1] - y_hat[row,currentCol])
        errSquaredSum += (y_real[row,1] - y_hat[row,currentCol])^2
    end
    meanErr[currentCol] = errSum/nRows
    RMSE[currentCol] = sqrt(errSquaredSum/nRows)
    Rsquare[currentCol] = 1- errSquaredSum/SSTO

    # Calculate insample error

    # Process data
end

tempArr = zeros(1,8)

classRate = findmax(classificationRate)[1]
classIndex = findmax(classificationRate)[2]
OOSR = findmax(Rsquare)[1]
OOSRIndex = findmax(Rsquare)[2]
meanErrVal = findmax(meanErr)[1]
meanErrIndex = findmax(meanErr)[2]
RMSEVal = findmax(RMSE)[1]
RMSEIndex = findmax(RMSE)[2]
tempArr[1,1] = findmax(classificationRate)[1]
tempArr[1,5] = findmax(classificationRate)[2]
tempArr[1,2] = findmax(Rsquare)[1]
tempArr[1,6] = findmax(Rsquare)[2]
tempArr[1,3] = findmax(meanErr)[1]
tempArr[1,7] = findmax(meanErr)[2]
tempArr[1,4] = findmax(RMSE)[1]
tempArr[1,8] = findmax(RMSE)[2]

println("Classification: ",findmax(classificationRate))
println("OoS R²: ",findmax(Rsquare))
println("Mean Absolute Error: ",findmin(meanErr))
println("RMSE: ",findmin(RMSE))
=#
