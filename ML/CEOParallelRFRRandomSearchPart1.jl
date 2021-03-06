using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Distributions
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

path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
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

srand(1)
modelConfig = zeros(amountOfModels, 4)
counter = 1
for l1 = 1:nGammas
    for l2 = 1:nGammas
        for l3 = 1:nGammas
            for l4 = 1:nGammas
                modelConfig[counter,:] = [rand(Uniform(1,100)) rand(Uniform(10,10000)) rand(Uniform(1,100)) rand(Uniform(1,10000))]
                counter += 1
            end
        end
    end
end


#Initialization of parameters
w1N = repeat([1/11], outer = 11) #1/N weights
gamma = 2.4 #risk aversion
validationPeriod = 5
PMatrix = zeros(nRows-trainingSize, amountOfModels)
return1NMatrix = zeros(nRows-trainingSize, amountOfModels)
returnCEOMatrix = zeros(nRows-trainingSize, amountOfModels)
returnPerfectMatrix = zeros(nRows-trainingSize)

bestModelAmount = 5
bestModelConfigs = zeros(bestModelAmount, 4)
bestModelIndexes = zeros(bestModelAmount)

weightsPerfect = zeros(nRows-trainingSize, 11)
weightsCEO     = zeros(nRows-trainingSize, 11, amountOfModels)

rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFR/"
weightsPerfect = Array{Float64}(CSV.read(path*"weightsPerfect.csv",header=false, datarow=1, nullable=false))
returnPerfectMatrix = Array{Float64}(CSV.read(path*"returnPerfectMatrix.csv", header=false, datarow=1, nullable=false))

println("Starting CEO Validation loop")
t=inputArg1
trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

for m = 1+(inputArg2*25):25+(inputArg2*25)#:amountOfModels
    expectedReturns = zeros(11)
    println(m-(inputArg2*25), " out of ", 25)

    #CHANGES
    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    betaArray, U = @time(runCEORFR(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma, rfRatesVec))

    ##PREVIOUS
    #betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma))

    expectedReturns[1:10] = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
    expectedReturns[11] = rfRates[t+trainingSize]
    #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
    valY = zeros(11)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    valY[11] = rfRates[t+trainingSize]
    #return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, OOSRow[1][1:10], valY)
    rfRatesVec = rfRates[t:(t+trainingSize-1)]
    trainX = hcat(trainingXArrays[1][:,1:10], rfRatesVec)
    Sigma =  cov(trainX)
    F = lufact(Sigma)
    U = F[:U]  #Cholesky factorization of Sigma

    return1N, returnCEO, wStar = performMVOptimizationRISK(expectedReturns, U, gamma, valY, valY)
    weightsCEO[t, 1:11, m]        = wStar
    return1NMatrix[t, m]       = return1N
    returnCEOMatrix[t, m]      = returnCEO
    returnPerfect = returnPerfectMatrix[t]
    println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
    PMatrix[t, m] = calculatePvalue(return1N, returnPerfect, returnCEO)
    #trackReturn(returnCEOTotal, returnCEO)
end

# Writing files to be used in CEOParallelPart2
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/CEO/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/CEORFR/"
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_weightsCEO.csv", weightsPerfect[t,:], ",") # do additional stuff here
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_return1NMatrix.csv", return1NMatrix[t,:], ",")
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_returnCEOMatrix.csv", returnCEOMatrix[t,:], ",")
writedlm(path*string(inputArg1)*"_"*string(inputArg2)*"_PMatrix.csv", PMatrix[t,:], ",")
