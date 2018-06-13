#Code written by Skipper af Rosenborg and Esben Bager
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

#Set path for dataloading
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
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

rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]

#Set parameter
nGammas = 5
standY = YArrays[1]
nRows = size(standY)[1]
amountOfModels = nGammas^4

#Initialization of parameters
w1N = repeat([1/11], outer = 11) #1/N weights
gamma = 2 #risk aversion
validationPeriod = 0 # Number of periods to do parameter training
PMatrix = zeros(nRows-trainingSize, amountOfModels)

bestModelAmount = 5
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

return1NMatrix = zeros(nRows-trainingSize)
returnCEOMatrix = zeros(nRows-trainingSize, bestModelAmount)
returnPerfectMatrix = zeros(nRows-trainingSize)

weightsPerfect = zeros(nRows-trainingSize, 11)
weightsCEO     = zeros(nRows-trainingSize, 11, bestModelAmount)
expectedReturnMatrix = zeros(nRows-trainingSize, 11)
forecastErrors = zeros(nRows-trainingSize, 11)

#Load perfect results
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/RFR/VIXTimeTA2.4/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFR/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))

#Set models to investigate
bestModelConfigs[1,:] = [1 0.5  0.25  1]
bestModelConfigs[2,:] = [1 0.5  0.25  2.5]
bestModelConfigs[3,:] = [1 0.5  0.25  10]
bestModelConfigs[4,:] = [1 0.5  0.25  50]
bestModelConfigs[5,:] = [1 0.5  0.25  100]

for i = 1:bestModelAmount
    bestModelIndexes[i] = i
end

inputArg1 = 0 #83 breaks the trainingSize
inputArg1 = parse(Int64,ARGS[1])
if 10+validationPeriod+(10*inputArg1) <= nRows-trainingSize-2
    println("Starting CEO Validation loop")
    for t=validationPeriod+1+(10*inputArg1):10+validationPeriod+(10*inputArg1)#(nRows-trainingSize-2)
        println("Time $t/50")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

        for m = 1:bestModelAmount
            expectedReturns = zeros(11)

            rfRatesVec = rfRates[t:(t+trainingSize-1)]

            #Build and solve CEO
            betaArray, U = @time(runCEORFR(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma, rfRatesVec))

            # Forecast next period
            expectedReturns[1:10] = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
            expectedReturns[11] = rfRates[t+trainingSize]
            if m == 1
                expectedReturnMatrix[t, 1:10] = (exp.(expectedReturns[1:10])-1)*100
                expectedReturnMatrix[t, 11] = (exp.(expectedReturns[11])-1)
            end

            valY = zeros(11)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            valY[11] = rfRates[t+trainingSize]

            #Set Sigma for MV
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
    end
elseif validationPeriod+(10*inputArg1) <= nRows-trainingSize-2
    for t=validationPeriod+1+(10*inputArg1):(nRows-trainingSize-2)
        println("Time $t/50")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

        for m = 1:bestModelAmount
            expectedReturns = zeros(11)
            rfRatesVec = rfRates[t:(t+trainingSize-1)]
            betaArray, U = @time(runCEORFR(trainingXArrays, trainingYArrays, bestModelConfigs[m, :], gamma, rfRatesVec))

            expectedReturns[1:10] = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
            expectedReturns[11] = rfRates[t+trainingSize]
            if m == 1
                expectedReturnMatrix[t, 1:10] = (exp.(expectedReturns[1:10])-1)*100
                expectedReturnMatrix[t, 11] = (exp.(expectedReturns[11])-1)
            end

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
            returnCEOMatrix[t, Int64(bestModelIndexes[m])]  = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            if m == 1
                forecastErrors[t, 1:11]  = forecastRow-expectedReturnMatrix[t,:]
            end
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, Int64(bestModelIndexes[m])] = calculatePvalue(return1N, returnPerfect, returnCEO)
        end
    end
end

#Output data
combinedPortfolios = hcat(returnPerfectMatrix[1:nRows-trainingSize-2, 1], return1NMatrix[1:nRows-trainingSize-2, 1],
    returnCEOMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)], PMatrix[1:nRows-trainingSize-2, Array{Int64}(bestModelIndexes)])
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFR/"
writedlm(path*string(inputArg1)*"_returnPvalueOutcome1to200.csv", combinedPortfolios, ",")
for i = 1:bestModelAmount
    writedlm(path*"wightsCEO_Model"*string(i)*"_"*string(inputArg1)*".csv",weightsCEO[:,:,i], ",")
end

combinedPortfolios2 = hcat(expectedReturnMatrix, forecastErrors)
writedlm(path*string(inputArg1)*"_forecastsAndErrors1to200.csv", combinedPortfolios2, ",")
println("Finished Everything")
