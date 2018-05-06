using StatsBase
using DataFrames
using CSV

#trainingSizeInput = parse(Int64, ARGS[1])
trainingSize = 240

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
noDurModel = [1 0 1 1 1]
testModel = [0 1 0 0 0]
modelMatrix[1, :] = noDurModel
for i=2:industriesTotal
    modelMatrix[i, :] = noDurModel
end
##START OF A METHOD

path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexDataDiff/"

XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

riskAversions = linspace(0, 2.4, 10)
XArrays, YArrays = generateXandYs(industries, modelMatrix)
gamma = 0.1 #regularization term in LASSO
nRows = size(XArrays[1])[1]
w1N = repeat([0.1], outer = 10) #1/N weights
return1NMatrix = zeros(nRows-trainingSize)
returnPPDMatrix = zeros(nRows-trainingSize)

weightsPPD = zeros(nRows-trainingSize, 10)
forecastRows = zeros(nRows-trainingSize, 10)

for g = 1:10
    fileName = "Results"
    gammaRisk = riskAversions[g] #riskAversion in MV optimization
    total = nRows-trainingSize-1
    for t = 1:(nRows-trainingSize-1)
        println("time $t / $total, gammaRisk $g / 10 ")
        trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
        expectedReturns = zeros(industriesTotal)
        for i = 1:industriesTotal
            ISRsquared, Indicator, estimate, bSolved = generatSolveAndProcess(trainingXArrays[i], trainingYArrays[i], validationXRows[i][1,:], validationY[i][1], gamma)
            expectedReturns[i] = estimate
        end
        expectedReturns
        Sigma =  cov(trainingXArrays[1][:,1:10])

        #A=U^(T)U where U is upper triangular with real positive diagonal entries
        F = lufact(Sigma)
        U = F[:U]  #Cholesky factorization of Sigma

        #getting the actual Y values for each industry
        valY = zeros(10)
        for i = 1:10
            valY[i] = validationY[i][1]
        end
        return1N, returnPPD, wPPD, forecastRow = performMVOptimization(expectedReturns, U, gammaRisk, valY, valY)
        weightsPPD[t, 1:10]    = wPPD
        forecastRows[t, 1:10]  = forecastRow
        return1NMatrix[t]      = return1N
        returnPPDMatrix[t]     = returnPPD
    end
    fileName = fileName*"_train"*string(trainingSize)*"_"*string(gammaRisk)
    writedlm(fileName*"ppdWeights.csv", weightsPPD,",")
    writedlm(fileName*"ppdReturns.csv", returnPPDMatrix,",")
    writedlm(fileName*"ppd1N.csv", return1NMatrix,",")
end


#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd(path)
println("Finished")
