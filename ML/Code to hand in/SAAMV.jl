#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV

#Set training size
#trainingSizeInput = parse(Int64, ARGS[1])
trainingSize = 12

#Set path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML/Lasso_Test"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"
@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Setup data
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
testModel = [0 1 0 0 0]
for i=2:industriesTotal
    modelMatrix[i, :] = testModel
end

#Set output path
#path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexDataDiff/"

#Load data
XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

riskAversions = linspace(0, 2.4, 10)
XArrays, YArrays = generateXandYs(industries, modelMatrix)

nRows = size(XArrays[1])[1]
w1N = repeat([0.1], outer = 10) #1/N weights
return1NMatrix = zeros(nRows-trainingSize)
returnSAAMatrix = zeros(nRows-trainingSize)

weightsSAA = zeros(nRows-trainingSize, 10)
forecastRows = zeros(nRows-trainingSize, 10)

startPoint = 241 #194608
endPoint = 1080 #201607



for g = 1:10
    fileName = "Results"
    gammaRisk = riskAversions[g] #riskAversion in MV optimization
    total = (endPoint-trainingSize)
    for t = (startPoint-trainingSize):(endPoint-trainingSize)
        println("time $t / $total, gammaRisk $g / 10 ")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
        expectedReturns = zeros(industriesTotal)
        #Get expected return as mean of past training data
        for i=1:10
            expectedReturns[i] = mean(trainingYArrays[i][:])
        end

        Sigma =  cov(trainingXArrays[1][:,1:10])

        #A=U^(T)U where U is upper triangular with real positive diagonal entries
        F = lufact(Sigma)
        U = F[:U]  #Cholesky factorization of Sigma

        #getting the actual Y values for each industry
        valY = zeros(10)
        for i = 1:10
            valY[i] = validationY[i][1]
        end
        return1N, returnSAA, wSAA, forecastRow = performMVOptimization(expectedReturns[:], U, gammaRisk, valY, valY)
        weightsSAA[t, 1:10]    = wSAA
        forecastRows[t, 1:10]  = forecastRow
        return1NMatrix[t]      = return1N
        returnSAAMatrix[t]     = returnSAA
    end
    fileName = fileName*"_train"*string(trainingSize)*"_"*string(gammaRisk)
    writedlm(fileName*"saaWeights.csv", weightsSAA,",")
    writedlm(fileName*"SAAReturns.csv", returnSAAMatrix,",")
    writedlm(fileName*"saa1N.csv", return1NMatrix,",")
end


#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd(path)
println("Finished")
