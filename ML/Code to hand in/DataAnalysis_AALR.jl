#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV
using JuMP
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Set parameters for data analysis
timeSpan = 240
if timeSpan == 12
    nRows = 1070
else
    nRows = 1060+24-timeSpan
end

#Define path to datafile
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/"*string(timeSpan)*"-1/"
cd(path)

mainData = zeros(3, nRows, 1472)

#Load 3 best models
for i = 1:nRows
    if timeSpan == 120
        f = open(string(i)*"AALRBestK_120.csv")
    elseif timeSpan == 240
        f = open(string(i)*"AALRBestK_240.csv")
    else
        f = open(string(i)*"AALRBestK.csv")
    end
    readline(f)
    s = readline(f)
    mainData[1, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    mainData[2, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    mainData[3, i, :] = parse.(Float64, split(s, ","))
end

stringArr = Array{String}(3)

#Define path to real values
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/"
cd(path)
y_realFull = CSV.read("RealValue.CSV",
    delim = ',', nullable=false, header=["Date", "y_real"], types = [Int64, Float64])

y_realTimeSpan = zeros(size(y_realFull)[1]-timeSpan)
for k = 1:size(y_realFull)[1]-timeSpan-1
    y_realTimeSpan[k] = mean(y_realFull[1:k+timeSpan-1,2])
end

tempArr = zeros(3,8)
for i = 1:3
    startIndex=find(x -> x == 194608, mainData[i,:,1])[1]

    if timeSpan == 12
        y_real = mainData[i,startIndex:end-2,3]
        y_hat = mainData[i,startIndex:end-2,4]
    else
        y_real = mainData[i,startIndex:end-4,3]
        y_hat = mainData[i,startIndex:end-4,4]
    end
    nRows=840

    #Initialise arrays
    classificationRate = Array{Float64}(3)
    meanErr = Array{Float64}(3)
    RMSE = Array{Float64}(3)
    Rsquare = Array{Float64}(3)

    SSTO = sum((y_real[k,1]-mean(y_real[:,1]))^2 for k=1:nRows)

    #Calculate classification rate
    trueClass = 0
    for row = 1:nRows
        if sign(y_real[row]) == sign(y_hat[row])
            trueClass += 1
        end
    end
    classificationRate[i] = trueClass/nRows

    #Initialise and calculate AME, RMSE & R^2
    errSum = 0
    errSquaredSum = 0
    SSres = 0
    SSTO = 0
    for row = 1:nRows
        SSres += (y_real[row,1] - y_hat[row])^2
        SSTO += (y_real[row,1]-y_realTimeSpan[row])^2
        errSum += abs(y_real[row,1] - y_hat[row])
        errSquaredSum += (y_real[row,1] - y_hat[row])^2
    end
    meanErr = errSum/nRows
    RMSE = sqrt(errSquaredSum/nRows)
    Rsquare = 1-SSres/SSTO

    # Find hte best results
    tempArr[i,1] = findmax(classificationRate)[1]
    tempArr[i,2] = findmax(Rsquare)[1]
    tempArr[i,3] = findmin(meanErr)[1]
    tempArr[i,4] = findmin(RMSE)[1]
    tempArr[i,5] = findmax(classificationRate)[2]
    tempArr[i,6] = findmax(Rsquare)[2]
    tempArr[i,7] = findmin(meanErr)[2]
    tempArr[i,8] = findmin(RMSE)[2]
end

# Output results
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/Summary "*string(timeSpan)
f = open(path*".csv", "w")
write(f, "Dataset, Classification Rate, R^2, Mean Error, RMSE, Classification Rate Index, R^2 Index, Mean Error Index, RMSE Index\n")
writecsv(f,hcat([1,2,3],tempArr))
close(f)
println("Done")


#Analysis of objective in SpeedTest analysis
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/Speed Benchmark Proffs/Final copy/"
testArr = [1, 101, 201, 301, 401, 501]
outArr = zeros(6,4)
for i = 1:6
    test1 = CSV.read(path*string(testArr[i])*"bestNorm.csv", delim="\t", types=fill(Float64,3),datarow = 1, header=false, nullable=false)
    test2 = CSV.read(path*string(testArr[i])*"bestWarm.csv", delim="\t", types=fill(Float64,3),datarow = 1, header=false, nullable=false)
    test3 = CSV.read(path*string(testArr[i])*"bestMath.csv", delim="\t", types=fill(Float64,3),datarow = 1, header=false, nullable=false)
    test4 = CSV.read(path*string(testArr[i])*"bestHeuristic.csv", delim="\t", types=fill(Float64,3),datarow = 1, header=false, nullable=false)

    println(indmin(Array(test1)),"\t",indmin(Array(test2)),"\t",indmin(Array(test3)),"\t",indmin(Array(test4)))

    outArr[i,1] = minimum(Array(test1))
    outArr[i,2] = minimum(Array(test2))
    outArr[i,3] = minimum(Array(test3))
    outArr[i,4] = minimum(Array(test4))
end
