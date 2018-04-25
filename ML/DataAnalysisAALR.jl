using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

nRows = 1070
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/12-1/"
cd(path)

mainData = zeros(3, nRows, 1472)

for i = 1:nRows
    f = open(string(i)*"AALRBestK.csv")
    readline(f)
    s = readline(f)
    mainData[1, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    mainData[2, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    mainData[3, i, :] = parse.(Float64, split(s, ","))
end

stringArr = Array{String}(3)

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/"
cd(path)
y_realFull = CSV.read("RealValue.CSV",
    delim = ',', nullable=false, header=["Date", "y_real"], types = [Int64, Float64])

timeSpan = 12
y_realTimeSpan = zeros(size(y_realFull)[1]-timeSpan-4)
for k = 1:size(y_realFull)[1]-timeSpan-4
    y_realTimeSpan[k] = mean(y_realFull[k:k+timeSpan-1,2])
end

tempArr = zeros(3,8)
for i = 1:3
    y_real = mainData[i,:,3]
    y_hat = mainData[i,:,4]

    classificationRate = Array{Float64}(3)
    meanErr = Array{Float64}(3)
    RMSE = Array{Float64}(3)
    Rsquare = Array{Float64}(3)

    SSTO = sum((y_real[k,1]-mean(y_real[:,1]))^2 for k=1:nRows)

    trueClass = 0
    for row = 1:nRows
        if sign(y_real[row]) == sign(y_hat[row])
            trueClass += 1
        end
    end
    classificationRate[i] = trueClass/nRows

    # SSTO for each i = (y_real (the actual one) - y_realTimeSpan)^2
    # R^2 for each i = squared err / SSTO

    indvRSquared = Array{Float64}(nRows,1)
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

    # Process data
    tempArr[i,1] = findmax(classificationRate)[1]
    tempArr[i,2] = findmax(Rsquare)[1]
    tempArr[i,3] = findmax(meanErr)[1]
    tempArr[i,4] = findmax(RMSE)[1]
    tempArr[i,5] = findmax(classificationRate)[2]
    tempArr[i,6] = findmax(Rsquare)[2]
    tempArr[i,7] = findmax(meanErr)[2]
    tempArr[i,8] = findmax(RMSE)[2]
end

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/Summary "*string(timeSpan)
f = open(path*".csv", "w")
write(f, "Dataset, Classification Rate, R^2, Mean Error, RMSE, Classification Rate Index, R^2 Index, Mean Error Index, RMSE Index\n")
writecsv(f,hcat([1,2,3],tempArr))
close(f)
