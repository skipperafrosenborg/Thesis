using StatsBase
using DataFrames
using CSV

#trainingSizeInput = parse(Int64, ARGS[1])
trainingSize = 120

#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML/Lasso_Test"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"
@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#=
VIX = 1
raw = 0
expTrans = 1
timeTrans = 1
TA = 1
trainingSize = 48
=#
possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
testModel = [0 1 0 0 0]
for i=1:industriesTotal
    modelMatrix[i, :] = testModel
end
##START OF A METHOD

path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexDataDiff/"

XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

riskAversions = linspace(0, 2.4, 10)
XArrays, YArrays = generateXandYs(industries, modelMatrix)

nRows = size(XArrays[1])[1]
w1N = repeat([0.1], outer = 11) #1/N weights
return1NMatrix = zeros(nRows-trainingSize)
returnSAAMatrix = zeros(nRows-trainingSize)

weightsSAA = zeros(nRows-trainingSize, 11)
forecastRows = zeros(nRows-trainingSize, 11)

rfRates = loadRiskFreeRate("NoDur", path)
rfRates = rfRates[:,1]
startPoint = 241 #194608
endPoint = 1080 #201607


for g = 1:10
    fileName = "Results"
    gammaRisk = riskAversions[g] #riskAversion in MV optimization
    total = endPoint-trainingSize
    for t = (startPoint-trainingSize):(endPoint-trainingSize)
        println("time $t / $total, gammaRisk $g / 10 ")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
        expectedReturns = zeros(industriesTotal+1)
        for i=1:10
            expectedReturns[i] = mean(trainingYArrays[i][:])
        end
        expectedReturns[11] = rfRates[t+trainingSize]
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
        return1N, returnSAA, wSAA, forecastRow = performMVOptimizationRISK(expectedReturns[:], U, gammaRisk, valY, valY)
        weightsSAA[t, 1:11]    = wSAA
        forecastRows[t, 1:11]  = forecastRow
        return1NMatrix[t]      = return1N
        returnSAAMatrix[t]     = returnSAA
    end
    fileName = fileName*"_train"*string(trainingSize)*"_"*string(gammaRisk)
    writedlm(fileName*"SAARFRWeights.csv", weightsSAA,",")
    writedlm(fileName*"SAARFRReturns.csv", returnSAAMatrix,",")
    #writedlm(fileName*"SAARFR1N.csv", return1NMatrix,",")
end


#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd(path)
println("Finished")
