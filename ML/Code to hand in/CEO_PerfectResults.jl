#Code written by Skipper af Rosenborg and Esben Bager
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

#Set path
#Esben's path
#path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexData"

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"

#Define initial parameters
trainingSize = 240
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

#Define datasets to load
modelMatrix = zeros(industriesTotal, possibilities)
noDurModel = [1 0 0 1 1]
modelMatrix[1, :] = noDurModel
for i=2:industriesTotal
    modelMatrix[i, :] = noDurModel
end

#Initialize arrays
XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

#Fill arrays with data
XArrays, YArrays = generateXandYs(industries, modelMatrix)

standY = YArrays[1]
nRows = size(standY)[1]

#Initialization of parameters
gamma = 0 #risk aversion
returnPerfectMatrix = zeros(nRows-trainingSize)
weightsPerfect = zeros(nRows-trainingSize, 10)

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

#Output results
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEO/"
writedlm(path*"weightsPerfect.csv", weightsPerfect, ",")
writedlm(path*"returnPerfectMatrix.csv", returnPerfectMatrix, ",")
