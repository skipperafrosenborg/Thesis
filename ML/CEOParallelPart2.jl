##### MEAN-VARIANCE OPTIMIZATION
#This is not supposed to produce great results, since mean and covariance
#is supposed to be very exact before good weights are found
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

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

trainingSize = 240
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
#noDurModel = [1 0 1 1 1]
#noDurModel = [1 0 1 0 1]
noDurModel = [0 1 0 0 0]
testModel = [0 1 0 0 0]
modelMatrix[1, :] = noDurModel
for i=2:industriesTotal
    modelMatrix[i, :] = noDurModel
end


XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

XArrays, YArrays = generateXandYs(industries, modelMatrix)

nGammas = 5
standY = YArrays[1]
SSTO = sum((standY[i]-mean(standY[:]))^2 for i=1:length(standY))
lambdaValues = log.(logspace(100, SSTO/2, nGammas))
nRows = size(standY)[1]
amountOfModels = nGammas^4


modelConfig = zeros(amountOfModels, 4)
counter = 1
for l1 = 1:nGammas
    for l2 = 1:nGammas
        for l3 = 1:nGammas
            for l4 = 1:nGammas
                modelConfig[counter,:] = [lambdaValues[l1] lambdaValues[l2] lambdaValues[l3] lambdaValues[l4]]
                counter += 1
            end
        end
    end
end
modelConfig

#Initialization of parameters
w1N = repeat([0.1], outer = 10) #1/N weights
gamma = 10 #risk aversion
validationPeriod = 5
PMatrix = zeros(nRows-trainingSize, amountOfModels)
return1NMatrix = zeros(nRows-trainingSize)
returnCEOMatrix = zeros(nRows-trainingSize, amountOfModels)
returnPerfectMatrix = zeros(nRows-trainingSize)

bestModelAmount = 5
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

weightsPerfect = zeros(nRows-trainingSize, 10)
weightsCEO     = zeros(nRows-trainingSize, 10)

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))

for i = 1:5
    for j = 0:24
        #path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
        path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
        PMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_PMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))
        return1NMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_return1NMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))
        returnCEOMatrixTemp = Array{Float64}(CSV.read(path*string(i)*"_"*string(j)*"_returnCEOMatrix.csv",header=false, datarow=1, nullable=false, types=[Float64]))

        PMatrix[i,:] += PMatrixTemp
        return1NMatrix[i] += return1NMatrixTemp[i]
        returnCEOMatrix[i,:] += returnCEOMatrixTemp
    end
end

#Load weightsPerfect and return perfect matrix
#Load PMatrix

modelMeans = mean(PMatrix, 1)
for i = 1:bestModelAmount
    bestModelIndex = indmax(modelMeans)
    bestModelConfigs[i,:] = modelConfig[bestModelIndex, :]
    bestModelIndexes[i] = bestModelIndex
    #place 0 at index place now and take new max as "next best model"
    modelMeans[bestModelIndex] = 0
end

inputArg1 = 83 #83 breaks the trainingSize
inputArg1 = parse(Int64,ARGS[1])
if 10+validationPeriod+(10*inputArg1) <= nRows-trainingSize-2
    println("Starting CEO Validation loop")
    for t=validationPeriod+1+(10*inputArg1):10+validationPeriod+(10*inputArg1)#(nRows-trainingSize-2)
        println("Time $t/50")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

        for m = 1:bestModelAmount
            betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma))
            expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

            #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
            valY = zeros(10)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
            weightsCEO[t, 1:10]     = wStar
            return1NMatrix[t]      = return1N
            returnCEOMatrix[t, Int64(bestModelIndexes[m])]     = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, Int64(bestModelIndexes[m])] = calculatePvalue(return1N, returnPerfect, returnCEO)
        end
    end
elseif validationPeriod+(10*inputArg1) <= nRows-trainingSize-2
    for t=validationPeriod+1+(10*inputArg1):(nRows-trainingSize-2)
        println("Time $t/50")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

        for m = 1:bestModelAmount
            betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma))
            expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

            #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
            valY = zeros(10)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
            weightsCEO[t, 1:10]     = wStar
            return1NMatrix[t]      = return1N
            returnCEOMatrix[t, Int64(bestModelIndexes[m])]     = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, Int64(bestModelIndexes[m])] = calculatePvalue(return1N, returnPerfect, returnCEO)
        end
    end
end

combinedPortfolios = hcat(returnPerfectMatrix[1:nRows-trainingSize-2, 1], return1NMatrix[1:nRows-trainingSize-2, 1],
    returnCEOMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)], PMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)])
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
writedlm(path*string(inputArg1)*"_returnPvalueOutcome1to200.csv", combinedPortfolios, ",")
println("Finished Everything")
