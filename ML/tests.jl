using StatsBase
using DataFrames
using CSV

#trainingSizeInput = parse(Int64, ARGS[1])
trainingSize = 12

#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML/Lasso_Test"
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"
@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

possibilities = 5
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
testModel = [0 1 0 0 0]
for i=2:industriesTotal
    modelMatrix[i, :] = testModel
end
##START OF A METHOD

path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexDataDiff/"

XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

riskAversions = linspace(0, 2.4, 10)
riskAversions = linspace(0, 4, 16)
XArrays, YArrays = generateXandYs(industries, modelMatrix)

nRows = size(XArrays[1])[1]
w1N = repeat([0.1], outer = 10) #1/N weights
return1NMatrix = zeros(nRows-trainingSize)
returnSAAMatrix = zeros(nRows-trainingSize)

weightsSAA = zeros(nRows-trainingSize, 10)
forecastRows = zeros(nRows-trainingSize, 10)

startPoint = 241 #194608
endPoint = 1080 #201607

g=1
fileName = "Results"
gammaRisk = riskAversions[g] #riskAversion in MV optimization
total = (endPoint-trainingSize)

t=1
println("time $t / $total, gammaRisk $g / 10 ")
trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
expectedReturns = zeros(industriesTotal)
for i=1:10
    expectedReturns[i] = mean(trainingYArrays[i][:])
end

Sigma =  cov(trainingXArrays[1][:,1:10])

a = w1N'*Sigma*w1N*4
b = w1N'*expectedReturns
a/b
#A=U^(T)U where U is upper triangular with real positive diagonal entries
F = lufact(Sigma)

U = F[:U]  #Cholesky factorization of Sigma

#getting the actual Y values for each industry
valY = zeros(10)
for i = 1:10
    valY[i] = validationY[i][1]
end
indexes = 10
M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
@variables M begin
        w[1:indexes]
        u[1:indexes]
        z
        y
end
gammaRisk=2.4
@objective(M,Min, gammaRisk*y - expectedReturns'*w)
@constraint(M, 0 .<= w)
@constraint(M, sum(w[i] for i=1:indexes) == 1)
@constraint(M, norm([2*U'*w;y-1]) <= y+1)
solve(M)
obj24 = getobjectivevalue(M)
wStar0 = getvalue(w)
wStar24 = getvalue(w)
