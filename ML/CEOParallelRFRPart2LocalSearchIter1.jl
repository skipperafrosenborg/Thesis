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
w1N = repeat([1/11], outer = 11) #1/N weights
gamma = 2.4 #risk aversion
validationPeriod = 0 # Number of periods to do parameter training
PMatrix = zeros(nRows-trainingSize-2, amountOfModels)

bestModelAmount = nGammas^3
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

return1NMatrix = zeros(nRows-trainingSize)
returnCEOMatrix = zeros(nRows-trainingSize, bestModelAmount)
returnPerfectMatrix = zeros(nRows-trainingSize)

weightsPerfect = zeros(nRows-trainingSize, 11)
weightsCEO     = zeros(nRows-trainingSize, 11, bestModelAmount)
expectedReturnMatrix = zeros(nRows-trainingSize, 11)
forecastErrors = zeros(nRows-trainingSize, 11)

rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/VIXTimeTA2.4/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFRSearch/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))

counter = 1
for l2 = 1:nGammas
    for l3 = 1:nGammas
        for l4 = 1:nGammas
            bestModelConfigs[counter,:] = [1 lambdaValues[l2] lambdaValues[l3] lambdaValues[l4]]
            counter += 1
        end
    end
end

futureLambdaValues = zeros(3,5)
futureLambdaValues[1,:] = lambdaValues
futureLambdaValues[2,:] = lambdaValues
futureLambdaValues[3,:] = lambdaValues

PMatrix = zeros(nRows-trainingSize-2, amountOfModels)
println("Lambda 2 ", futureLambdaValues[1,:])
println("Lambda 3 ", futureLambdaValues[2,:])
println("Lambda 4 ", futureLambdaValues[3,:])
for j = 1:parse(Int64,"4")#ARGS[2])
    PMatrix = zeros(nRows-trainingSize-2, amountOfModels)
    println(j)
    for i = 741:790
        #path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFRSearch/"
        path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/CEORFRSearch/740-790/"
        PMatrixTemp = Array{Float64}(CSV.read(path*"Iter"*string(j-1)*"/"*string(i-1)*"_returnPvalueOutcome1to200.csv",header=false, datarow=1, nullable=false, types=fill(Float64,252)))

        PMatrix[i,:] += PMatrixTemp[i,3+125:end]
    end
    bestIndex = indmax(mean(PMatrix[:,:],1))
    bestConfig = bestModelConfigs[bestIndex,:]

    for k = 1:3
        bestIndx = find(x -> bestConfig[k+1] == x,futureLambdaValues[k,:])[1]

        best = futureLambdaValues[k,bestIndx]

        if bestIndx == 1
            min = futureLambdaValues[k,1]*0
            max = futureLambdaValues[k,2]
        elseif bestIndx == 5
            min = futureLambdaValues[k,4]
            max = futureLambdaValues[k,5]*2
        else
            min = futureLambdaValues[k,bestIndx-1]
            max = futureLambdaValues[k,bestIndx+1]
        end

        futureLambdaValues[k,1] = best-(best-min)*0.8
        futureLambdaValues[k,2] = best-(best-min)*0.2
        futureLambdaValues[k,3] = best
        futureLambdaValues[k,4] = best+(max-best)*0.2
        futureLambdaValues[k,5] = best+(max-best)*0.8
    end
    println("Lambda 2 ", futureLambdaValues[1,:])
    println("Lambda 3 ", futureLambdaValues[2,:])
    println("Lambda 4 ", futureLambdaValues[3,:])

    counter = 1
    for l2 = 1:nGammas
        for l3 = 1:nGammas
            for l4 = 1:nGammas
                bestModelConfigs[counter,:] = [1 futureLambdaValues[1,l2] futureLambdaValues[2,l3] futureLambdaValues[3,l4]]
                counter += 1
            end
        end
    end
end

println(bestModelConfigs[1:10,:])

#Load weightsPerfect and return perfect matrix
#Load PMatrix

#modelMeans = mean(PMatrix, 1)


for i = 1:bestModelAmount
	bestModelIndexes[i]=i
end

PMatrix = zeros(nRows-trainingSize-2, amountOfModels)
inputArg1 = 0 #83 breaks the trainingSize
inputArg1 = parse(Int64,ARGS[1])
println("Starting CEO Validation loop")
t=1+inputArg1
println("Time $t/50")
trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

for m = 1:bestModelAmount
    expectedReturns = zeros(11)
    #CHANGES
    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    betaArray, U = @time(runCEORFR(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma, rfRatesVec))

    #betaArrayCopy = betaArray
    ##PREVIOUS
    #betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma))
    expectedReturns[1:10] = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
    expectedReturns[11] = rfRates[t+trainingSize]
    if m == 1
        expectedReturnMatrix[t, 1:10] = (exp.(expectedReturns[1:10])-1)*100
        expectedReturnMatrix[t, 11] = (exp.(expectedReturns[11])-1)
    end
    #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
    valY = zeros(11)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    valY[11] = rfRates[t+trainingSize]

    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    trainX = hcat(trainingXArrays[1][:,1:10], rfRatesVec)
    Sigma =  cov(trainX)
    F = lufact(Sigma)
    U = F[:U]  #Cholesky factorization of Sigma

    return1N, returnCEO, wStar, forecastRow = performMVOptimizationRISK(expectedReturns, U, gamma, valY, valY)
    weightsCEO[t, 1:11, Int64(m)] = wStar
    return1NMatrix[t]      = return1N
    returnCEOMatrix[t, Int64(bestModelIndexes[m])] = returnCEO
    returnPerfect = returnPerfectMatrix[t]
    if m == 1
        forecastErrors[t, 1:11]  = forecastRow-expectedReturnMatrix[t,:]
    end
    println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
    PMatrix[t, Int64(bestModelIndexes[m])] = calculatePvalue(return1N, returnPerfect, returnCEO)
end

combinedPortfolios = hcat(returnPerfectMatrix[1:nRows-trainingSize-2, 1], return1NMatrix[1:nRows-trainingSize-2, 1],
    returnCEOMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)], PMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)])
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFRSearch/Iter"*ARGS[2]*"/"
writedlm(path*string(inputArg1)*"_returnPvalueOutcome1to200.csv", combinedPortfolios, ",")
for i = 1:bestModelAmount
    writedlm(path*"wightsCEO_Model"*string(i)*"_"*string(inputArg1)*".csv",weightsCEO[:,:,i], ",")
end

combinedPortfolios2 = hcat(expectedReturnMatrix, forecastErrors)
writedlm(path*string(inputArg1)*"_forecastsAndErrors1to200.csv", combinedPortfolios2, ",")
println("Finished Everything")
