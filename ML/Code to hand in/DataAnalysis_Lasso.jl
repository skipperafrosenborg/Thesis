#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

#Initilise parameters
industry = "NoDur"
folder = "240"
industryArr = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"
for industry = industryArr
    for i = [12, 24, 36, 48, 120, 240]
        println("Industry ",industry," time period ",i)
        path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"
        folder = string(i)
        #Write a summary file for given industry and training period
        writeFile()
    end
end


path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"

stringArr = Array{String}(10)
for j = 1:10
    stringArr[j] = "gamma =$j"
end

function logStuff(VIX, raw, timeTrans, expTrans, TA, Top25)
    timeSpan = parse(Int64,folder)
    tempString = ""
    #Define datasat based on input
    if raw == 0
        if VIX ==1
            tempString = tempString*"VIX_Macro"
        else
            tempString = tempString*"Macro"
        end
    else
        tempString = tempString*"Raw"
    end

    if timeTrans == 1
        tempString = tempString*"Time"
    end

    if expTrans == 1
        tempString = tempString*"Exp"
    end

    if TA == 1
        tempString = tempString*"TA"
    end

    #Load real values
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"
    cd(path)
    y_realFull = CSV.read("RealValue.CSV",
        delim = ',', nullable=false, header=["Date", "y_real"], types = [Int64, Float64])

    y_realTimeSpan = zeros(size(y_realFull)[1]-timeSpan)
    for k = 1:size(y_realFull)[1]-timeSpan-1
        y_realTimeSpan[k] = mean(y_realFull[1:k+timeSpan-1,2])
    end

    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"
    cd(path)

    y_real = CSV.read(folder*"_"*tempString*"_real.CSV", delim = ',', nullable = false, header = ["Iteration",
        "Date", "Reseccion", "Real Value"], types =[Float64, Int64, Int64, Float64], datarow=2)

    #If load a dataset based on top250
    if Top25 == 1 &&timeSpan == 240
        y_hat = CSV.read(folder*"_"*tempString*"_over25PercentPredictionTermspredicted.CSV",
            delim = ',', nullable=false, header=vcat("Iteration","Date","Recession",stringArr), types=[Float64, Int64, Int64, Float64,
            Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64], datarow=2)
        tempString = tempString*"_over25PercentPredictionTerms"
    else
        y_hat = CSV.read(folder*"_"*tempString*"_predicted.CSV",
            delim = ',', nullable=false, header=vcat("Iteration","Date","Recession",stringArr), types=[Float64, Int64, Int64, Float64,
            Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64], datarow=2)
    end

    #Make sure we start at same place
    startIndex=find(x -> x == 194608, y_real[:,2])[1]

    #Trim datasets
    y_realTimeSpan = y_realTimeSpan[startIndex:end-6,end]
    y_real = y_real[startIndex:end-5,4:end]
    y_hat = y_hat[startIndex:end-5,:4:end]

    nRows = size(y_hat)[1]
    nCols = size(y_hat)[2]

    #Initilise arrays
    classificationRate = Array{Float64}(nCols)
    meanErr = Array{Float64}(nCols)
    RMSE = Array{Float64}(nCols)
    Rsquare = Array{Float64}(nCols)

    #Calcualte CR, AME, RMSE, R^2 for each lambda value
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
        indvRSquared = Array{Float64}(nRows,1)
        errSum = 0
        errSquaredSum = 0
        SSres = 0
        SSTO = 0
        for row = 1:nRows
            SSres += (y_real[row,1] - y_hat[row,currentCol])^2
            SSTO += (y_real[row,1]-y_realTimeSpan[row])^2
            errSum += abs(y_real[row,1] - y_hat[row,currentCol])
            errSquaredSum += (y_real[row,1] - y_hat[row,currentCol])^2
        end
        meanErr[currentCol] = errSum/nRows
        RMSE[currentCol] = sqrt(errSquaredSum/nRows)
        Rsquare[currentCol] = 1-SSres/SSTO
    end

    #Output to logarray
    tempArr = zeros(1,8)

    classRate = findmax(classificationRate)[1]
    classIndex = findmax(classificationRate)[2]
    OOSR = findmax(Rsquare)[1]
    OOSRIndex = findmax(Rsquare)[2]
    meanErrVal = findmin(meanErr)[1]
    meanErrIndex = findmin(meanErr)[2]
    RMSEVal = findmin(RMSE)[1]
    RMSEIndex = findmin(RMSE)[2]
    tempArr[1,1] = findmax(classificationRate)[1]
    tempArr[1,2] = findmax(Rsquare)[1]
    tempArr[1,3] = findmin(meanErr)[1]
    tempArr[1,4] = findmin(RMSE)[1]
    tempArr[1,5] = findmax(classificationRate)[2]
    tempArr[1,6] = findmax(Rsquare)[2]
    tempArr[1,7] = findmin(meanErr)[2]
    tempArr[1,8] = findmin(RMSE)[2]
    #println("Classification: ",findmax(classificationRate))
    #println("OoS R²: ",findmax(Rsquare))
    #println("Mean Absolute Error: ",findmin(meanErr))
    #println("RMSE: ",findmin(RMSE))

    return tempArr, tempString
end

function writeFile()
    counterIter = 1
    logArr = zeros(25,8)
    stringToAppend = Array{String}(25,1)

    #Loop over datasets without macro prediction terms
    for timeTrans = 0:1
    	for expTrans = 0:1
    		for TA = 0:1
    			logArr[counterIter,:], stringToAppend[counterIter,1] = logStuff(0, 1, timeTrans, expTrans, TA, 0)
                counterIter += 1
    		end
    	end
    end

    #Loop over datasets with macro prediction terms
    for VIX = 0:1
    	for timeTrans = 0:1
    		for expTrans = 0:1
    			for TA = 0:1
    				logArr[counterIter,:], stringToAppend[counterIter,1]= logStuff(VIX, 0, timeTrans, expTrans, TA, 0)
                    counterIter += 1
    			end
    		end
    	end
    end

    logArr[counterIter,:], stringToAppend[counterIter, 1] = logStuff(1, 0, 1, 0, 1, 1)

    #Output data
    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/Summary New"*folder
    f = open(path*".csv", "w")
    write(f, "Dataset, Classification Rate, R^2, Mean Error, RMSE, Classification Rate Index, R^2 Index, Mean Error Index, RMSE Index\n")
    writecsv(f,hcat(stringToAppend,logArr))
    close(f)
end
