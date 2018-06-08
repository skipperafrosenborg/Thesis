##### MEAN-VARIANCE OPTIMIZATION
#This is not supposed to produce great results, since mean and covariance
#is supposed to be very exact before good weights are found

#Load packages and support functions
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
include("CEOSupportFunctions.jl")
println("Leeeeroooy Jenkins")
#Esben's path
#path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexData"

#Set path for dataloading
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

#Set parameters
trainingSize = 240
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

#Define the input and transformations
modelMatrix = zeros(industriesTotal, possibilities)
noDurModel = [1 0 0 1 1]
testModel = [1 0 0 1 1]
modelMatrix[1, :] = noDurModel
for i=2:industriesTotal
    modelMatrix[i, :] = noDurModel
end

#Create and fill X and Y arrays
XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)
XArrays, YArrays = generateXandYs(industries, modelMatrix)

#Set
nGammas = 5
standY = YArrays[1]
SSTO = sum((standY[i]-mean(standY[:]))^2 for i=1:length(standY))
lambdaValues = [0.1, 0.5, 1, 2, 10]
#lambda4Values = [1e4,1e5,1e6,1e7,1e8]
nRows = size(standY)[1]
amountOfModels = nGammas^3

modelConfig = zeros(amountOfModels, 4)
counter = 1
for l2 = 1:nGammas
    for l3 = 1:nGammas
        for l4 = 1:nGammas
            modelConfig[counter,:] = [1 lambdaValues[l2] lambdaValues[l3] lambdaValues[l4]]
            counter += 1
        end
    end
end
modelConfig

#Initialization of parameters
w1N = repeat([1/10], outer = 10) #1/N weights
gamma = 2.4 #risk aversion
validationPeriod = 0 # Number of periods to do parameter training
PMatrix = zeros(nRows-trainingSize-2, amountOfModels)

bestModelAmount = nGammas^3
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

return1NMatrix = zeros(nRows-trainingSize)
returnCEOMatrix = zeros(nRows-trainingSize, bestModelAmount)
returnPerfectMatrix = zeros(nRows-trainingSize)

weightsPerfect = zeros(nRows-trainingSize, 10)
weightsCEO     = zeros(nRows-trainingSize, 10, bestModelAmount)
expectedReturnMatrix = zeros(nRows-trainingSize, 10)
forecastErrors = zeros(nRows-trainingSize, 10)

#= comment out as we run with no risk free option
rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]
=#

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/VIXTimeTA2.4/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))
#=
for i = 1:5
    for j = 0:24
        #path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
        #path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFR/"
        PMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_PMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))
        return1NMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_return1NMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))
        returnCEOMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_returnCEOMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))

        PMatrix[i,:] += PMatrixTemp
        return1NMatrix[i] += return1NMatrixTemp[i]
        returnCEOMatrix[i,:] += returnCEOMatrixTemp
    end
end
=#
#Load weightsPerfect and return perfect matrix
#Load PMatrix

#modelMeans = mean(PMatrix, 1)
counter = 1
for l2 = 1:nGammas
    for l3 = 1:nGammas
        for l4 = 1:nGammas
            bestModelConfigs[counter,:] = [1 lambdaValues[l2] lambdaValues[l3] lambdaValues[l4]]
            counter += 1
        end
    end
end

for i = 1:bestModelAmount
	bestModelIndexes[i]=i
end

inputArg1 = 0 #83 breaks the trainingSize
inputArg1 = parse(Int64,ARGS[1])
println("Starting CEO Validation loop")
t=1+inputArg1
println("Time $t/50")
trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

for m = 1:bestModelAmount
    expectedReturns = zeros(10)
    #CHANGES
    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma))

    #betaArrayCopy = betaArray
    ##PREVIOUS
    #betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma))
    expectedReturns[1:10] = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
    #expectedReturns[11] = rfRates[t+trainingSize]
    if m == 1
        expectedReturnMatrix[t, 1:10] = (exp.(expectedReturns[1:10])-1)*100
        #expectedReturnMatrix[t, 11] = (exp.(expectedReturns[11])-1)
    end
    #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
    valY = zeros(10)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    #valY[11] = rfRates[t+trainingSize]

    #rfRatesVec = rfRates[t:(t+trainingSize-1)]
    trainX = trainingXArrays[1][:,1:10]
    Sigma =  cov(trainX)
    F = lufact(Sigma)
    U = F[:U]  #Cholesky factorization of Sigma

    return1N, returnCEO, wStar, forecastRow = performMVOptimization(expectedReturns, U, gamma, valY, valY)
    weightsCEO[t, 1:10, Int64(m)] = wStar
    return1NMatrix[t]      = return1N
    returnCEOMatrix[t, Int64(bestModelIndexes[m])] = returnCEO
    returnPerfect = returnPerfectMatrix[t]
    if m == 1
        forecastErrors[t, 1:10]  = forecastRow-expectedReturnMatrix[t,:]
    end
    println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
    PMatrix[t, Int64(bestModelIndexes[m])] = calculatePvalue(return1N, returnPerfect, returnCEO)
end

combinedPortfolios = hcat(returnPerfectMatrix[1:nRows-trainingSize-2, 1], return1NMatrix[1:nRows-trainingSize-2, 1],
    returnCEOMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)], PMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)])
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEOSearch/Iter0/"
writedlm(path*string(inputArg1)*"_returnPvalueOutcome1to200.csv", combinedPortfolios, ",")
for i = 1:bestModelAmount
    writedlm(path*"wightsCEO_Model"*string(i)*"_"*string(inputArg1)*".csv",weightsCEO[:,:,i], ",")
end

combinedPortfolios2 = hcat(expectedReturnMatrix, forecastErrors)
writedlm(path*string(inputArg1)*"_forecastsAndErrors1to200.csv", combinedPortfolios2, ",")
println("Finished Everything")
