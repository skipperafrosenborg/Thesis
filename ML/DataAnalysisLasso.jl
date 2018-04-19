using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

folder = "12"
for i = [12, 24, 36, 48, 120, 240]
    folder = string(i)
    writeFile()
end

industry = "Enrgy"
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"

stringArr = Array{String}(10)
for j = 1:10
    stringArr[j] = "gamma =$j"
end

function logStuff(VIX, raw, timeTrans, expTrans, TA)
    timeSpan = parse(Int64,folder)
    tempString = ""
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

    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"
    cd(path)
    y_realFull = CSV.read("RealValue.CSV",
        delim = ',', nullable=false, header=["Date", "y_real"], types = [Int64, Float64])

    y_realTimeSpan = zeros(size(y_realFull)[1]-timeSpan)
    for k = 1:size(y_realFull)[1]-timeSpan-1
        y_realTimeSpan[k] = mean(y_realFull[k:k+timeSpan-1,2])
    end

    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/"*folder*"-1/"
    cd(path)

    y_real = CSV.read(folder*"_"*tempString*"_real.CSV", delim = ',', nullable = false, header = ["Iteration",
        "Date", "Reseccion", "Real Value"], types =[Float64, Int64, Int64, Float64], datarow=2)

    y_hat = CSV.read(folder*"_"*tempString*"_predicted.CSV",
        delim = ',', nullable=false, header=vcat("Iteration","Date","Recession",stringArr), types=[Float64, Int64, Int64, Float64,
        Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64], datarow=2)

    y_realTimeSpan = y_realTimeSpan[1:end-1,end]
    y_real = y_real[1:end,4:end]
    y_hat = y_hat[1:end,:4:end]

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

        # Fix R^2, skal den regnes som gennemsnit over alle de individuelt predicted R^2?
        # Gennemgå statistiske mål sammen med Esben

        # SSTO for each i = (y_real (the actual one) - y_realTimeSpan)^2
        # R^2 for each i = squared err / SSTO

        # Mean Error rate
        # Calculate RMSE
        # Calculate R^2
        indvRSquared = Array{Float64}(nRows,1)
        errSum = 0
        errSquaredSum = 0
        SSres = 0
        SSTO = 0
        for row = 1:nRows
            #indvRSquared[row] = 1-((y_real[row,1] - y_hat[row,currentCol])^2/(y_real[row,1]-y_realTimeSpan[row])^2)
            SSres += (y_real[row,1] - y_hat[row,currentCol])^2
            SSTO += (y_real[row,1]-y_realTimeSpan[row])^2
            errSum += abs(y_real[row,1] - y_hat[row,currentCol])
            errSquaredSum += (y_real[row,1] - y_hat[row,currentCol])^2
        end
        meanErr[currentCol] = errSum/nRows
        RMSE[currentCol] = sqrt(errSquaredSum/nRows)
        Rsquare[currentCol] = 1-SSres/SSTO
        #Rsquare[currentCol] = mean(indvRSquared)
        #println((1-errSquaredSum/SSTO))
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

    return tempArr, tempString
end

function writeFile()
    counterIter = 1
    logArr = zeros(24,8)
    stringToAppend = Array{String}(24,1)

    for timeTrans = 0:1
    	for expTrans = 0:1
    		for TA = 0:1
    			logArr[counterIter,:], stringToAppend[counterIter,1] = logStuff(0, 1, timeTrans, expTrans, TA)
                counterIter += 1
    		end
    	end
    end

    for VIX = 0:1
    	for timeTrans = 0:1
    		for expTrans = 0:1
    			for TA = 0:1
    				logArr[counterIter,:], stringToAppend[counterIter,1]= logStuff(VIX, 0, timeTrans, expTrans, TA)
                    counterIter += 1
    			end
    		end
    	end
    end


    path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"*industry*"/Summary "*folder
    f = open(path*".csv", "w")
    write(f, "Dataset, Classification Rate, R^2, Mean Error, RMSE, Classification Rate Index, R^2 Index, Mean Error Index, RMSE Index\n")
    writecsv(f,hcat(stringToAppend,logArr))
    close(f)
end
