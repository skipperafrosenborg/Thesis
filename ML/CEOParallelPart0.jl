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

inputArg1 = 1
inputArg2 = 1

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

trainingSize = 240
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
#noDurModel = [1 0 1 1 1]
#noDurModel = [1 0 1 0 1]
noDurModel = [1 0 0 1 1]
testModel = [1 0 0 1 1]
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

#Initialization of parameters
w1N = repeat([0.1], outer = 10) #1/N weights
gamma = 0 #risk aversion
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

#Establishing perfect results in order to avoid doing same mean-variance calculation over and over
for t=1:(nRows-trainingSize-2)
    trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
    valY = zeros(10)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    trainX = hcat(trainingXArrays[1][:,1:10], rfRatesVec)
    weightsPerfect[t, :], returnPerfectMatrix[t] = findPerfectResults(trainX, valY, valY, gamma)
end

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
writedlm(path*"weightsPerfect.csv", weightsPerfect, ",")
writedlm(path*"returnPerfectMatrix.csv", returnPerfectMatrix, ",")
