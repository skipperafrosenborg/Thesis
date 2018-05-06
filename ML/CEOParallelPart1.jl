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

#inputArg1 = 1   # should range from 1:5
#inputArg2 = 0   # should range from 0:24
inputArg1 = parse(Int64, ARGS[1])
inputArg2 = parse(Int64, ARGS[2])

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
return1NMatrix = zeros(nRows-trainingSize, amountOfModels)
returnCEOMatrix = zeros(nRows-trainingSize, amountOfModels)
returnPerfectMatrix = zeros(nRows-trainingSize)

bestModelAmount = 5
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

weightsPerfect = zeros(nRows-trainingSize, 10)
weightsCEO     = zeros(nRows-trainingSize, 10, nGammas^4)

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))

println("Starting CEO Validation loop")
t=inputArg1
trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

for m = 1+(inputArg2*25):25+(inputArg2*25)#:amountOfModels
    println(m-(inputArg2*25), " out of ", 25)
    betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma))
    expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

    #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
    valY = zeros(10)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    #return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, OOSRow[1][1:10], valY)
    return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
    weightsCEO[t, 1:10, m]        = wStar
    return1NMatrix[t, m]       = return1N
    returnCEOMatrix[t, m]      = returnCEO
    returnPerfect = returnPerfectMatrix[t]
    println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
    PMatrix[t, m] = calculatePvalue(return1N, returnPerfect, returnCEO)
    #trackReturn(returnCEOTotal, returnCEO)
end

# Writing files to be used in CEOParallelPart2
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_weightsCEO.csv", weightsPerfect[t,:], ",") # do additional stuff here
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_return1NMatrix.csv", return1NMatrix[t,:], ",")
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_returnCEOMatrix.csv", returnCEOMatrix[t,:], ",")
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_PMatrix.csv", PMatrix[t,:], ",")
