#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV

#Set trainingSize
#trainingSizeInput = parse(Int64, ARGS[1])
trainingSize = 240

#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML/Lasso_Test"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"

@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Set parameters
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
testModel = [1 0 0 1 1]
for i=1:industriesTotal
    modelMatrix[i, :] = testModel
end

#Initialise and load data
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexDataDiff/"
XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

riskAversions = linspace(0, 4, 16)
XArrays, YArrays = generateXandYs(industries, modelMatrix)
gamma = 0.1 #regularization term in LASSO
nRows = size(XArrays[1])[1]
w1N = repeat([0.1], outer = 11) #1/N weights
return1NMatrix = zeros(nRows-trainingSize)
returnPPDMatrix = zeros(nRows-trainingSize)

weightsPPD = zeros(nRows-trainingSize, 11)
forecastRows = zeros(nRows-trainingSize, 11)

expectedReturnMatrix = zeros(nRows-trainingSize, 11)
forecastErrors = zeros(nRows-trainingSize, 11)

rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]

#Generate perfect results with risk free rate
returnPerfectMatrix = zeros(nRows-trainingSize)
weightsPerfect = zeros(nRows-trainingSize, 11)
fileName = "/zhome/9f/d/88706/SpecialeCode/Results/MV/PointPrediction/"
for g = 1:10
    fileName = "PerfectRFR"
    gammaRisk = riskAversions[g]
    for t=1:840
        println("time $t / 840, gammaRisk $g / 10 ")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
        valY = zeros(11)
        for i = 1:10
            valY[i] = validationY[i][1]
        end
        valY[11] = rfRates[t+trainingSize]
        rfRatesVec = rfRates[t:(t+trainingSize-1)]
        trainX = hcat(trainingXArrays[1][:,1:10], rfRatesVec)
        weightsPerfect[t, :], returnPerfectMatrix[t] = findPerfectRFRResults(trainX, valY, valY, gammaRisk)
    end
    fileName = fileName*"_train"*string(trainingSize)*"_"*string(gammaRisk)
    writedlm(fileName*"Weights.csv", weightsPerfect,",")
    writedlm(fileName*"Returns.csv", returnPerfectMatrix,",")
end

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/LassoTest/"
path = "/zhome/9f/d/88706/SpecialeCode/Results/IndexData/LassoTest/"

for g = 1:16
    fileName = "/zhome/9f/d/88706/SpecialeCode/Results/MV/PointPrediction/"
    gammaRisk = riskAversions[g] #riskAversion in MV optimization
    total = nRows-trainingSize-1
    for t = 1:840
        println("time $t / $total, gammaRisk $g / 10 ")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
        expectedReturns = zeros(industriesTotal+1)
        #Load best dataset based on R^2 value
        for i = 1:industriesTotal
            summary = CSV.read(path*industries[i]*"/"*"Summary "*string(trainingSize)*".csv", nullable=false)

            maxR2, fileNameIndex = findmax(summary[:,3]) # Find file name for  max R^2

            bestFileName = summary[fileNameIndex,1]
            fileIndex = summary[fileNameIndex,7] # Find gamma

            estimate = CSV.read(path*industries[i]*"/"*string(trainingSize)*"-1/"*"240_"*bestFileName*"_predicted.CSV",nullable=false)
            expectedReturns[i] = estimate[:,3+fileIndex][t]
        end

        expectedReturns[11] = rfRates[t+trainingSize]

        expectedReturnMatrix[t, 1:10] = (exp.(expectedReturns[1:10])-1)*100
        expectedReturnMatrix[t, 11]   = exp(expectedReturns[11])-1

        rfRatesVec = rfRates[t:(t+trainingSize-1)]
        trainX = hcat(trainingXArrays[1][:,1:10], rfRatesVec)
        Sigma =  cov(trainX)

        #A=U^(T)U where U is upper triangular with real positive diagonal entries
        F = lufact(Sigma)
        U = F[:U]  #Cholesky factorization of Sigma

        #getting the actual Y values for each industry
        valY = zeros(11)
        for i = 1:10
            valY[i] = validationY[i][1]
        end
        valY[11] = rfRates[t+trainingSize]
        return1N, returnPPD, wPPD, forecastRow = performMVOptimizationRISK(expectedReturns, U, gammaRisk, valY, valY)
        weightsPPD[t, 1:11]    = wPPD
        forecastErrors[t, 1:11]  = forecastRow-expectedReturnMatrix[t,:]
        return1NMatrix[t]      = return1N
        returnPPDMatrix[t]     = returnPPD
    end
    fileName = fileName*"_train"*string(trainingSize)*"_"*string(gammaRisk)
    writedlm(fileName*"ppdRFRWeights.csv", weightsPPD,",")
    writedlm(fileName*"ppdRFRReturns.csv", returnPPDMatrix,",")
    writedlm(fileName*"ppdRFRErrors.csv", forecastErrors,",")
    writedlm(fileName*"ppdRFRForecasts.csv", expectedReturnMatrix,",")
end

println("Finished")
