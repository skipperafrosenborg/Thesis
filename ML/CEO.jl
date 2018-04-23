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
println("Leeeeroooy Jenkins")
#Esben's path
path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexData"

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexData/"
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexData/"




trainingSize = 240
possibilities = 5
industries = ["NoDur", "Durbl", "Enrgy", "HiTec", "Hlth", "Manuf", "Other", "Shops", "Telcm", "Utils"]
industriesTotal = length(industries)

modelMatrix = zeros(industriesTotal, possibilities)
noDurModel = [1 0 1 1 1]
testModel = [0 1 0 0 0]
modelMatrix[1, :] = noDurModel
for i=2:industriesTotal
    modelMatrix[i, :] = testModel
end


XArrays = Array{Array{Float64, 2}}(industriesTotal)
YArrays = Array{Array{Float64, 2}}(industriesTotal)

XArrays, YArrays = generateXandYs(industries, modelMatrix)

nGammas = 5
standY = YArrays[1]
SSTO = sum((standY[i]-mean(standY[:]))^2 for i=1:length(standY))
lambdaValues = log.(logspace(0, SSTO/2, nGammas))
nRows = size(standY)[1]
amountOfModels = nGammas^4
optError = zeros(nRows-trainingSize, amountOfModels)


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

nRows
trainingSize

a
for t=1:(nRows-trainingSize-2)
    trainingXArrays = Array{Array{Float64, 2}}(industriesTotal)
    trainingYArrays = Array{Array{Float64, 2}}(industriesTotal)

    validationXRows = Array{Array{Float64, 2}}(industriesTotal)
    validationY     = Array{Array{Float64, 2}}(industriesTotal)

    OOSXArrays      = Array{Array{Float64, 2}}(industriesTotal)
    OOSYArrays      = Array{Array{Float64, 2}}(industriesTotal)

    OOSRow          = Array{Array{Float64, 2}}(industriesTotal)
    OOSY            = Array{Array{Float64, 2}}(industriesTotal)
    for i = 1:industriesTotal
        trainingXArrays[i] = XArrays[i][t:(t-1+trainingSize), :]
        trainingYArrays[i] = YArrays[i][t:(t-1+trainingSize), :]

        validationXRows[i] = XArrays[i][(t+trainingSize):(t+trainingSize), :]
        validationY[i]     = YArrays[i][(t+trainingSize):(t+trainingSize), :]

        OOSXArrays[i]      = XArrays[i][(t+1):(t+trainingSize), :]
        OOSYArrays[i]      = YArrays[i][(t+1):(t+trainingSize), :]

        OOSRow[i]          = XArrays[i][(t+trainingSize+1):(t+trainingSize+1), :]
        OOSY[i]            = YArrays[i][(t+trainingSize+1):(t+trainingSize+1), :]
    end

    for m = 1:amountOfModels
        #errorArray, betaArray = runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :])
        
        ##Might need some sampling here to have a CDF
        ##Create forecast through validationXRows
        #
    end


end



modelConfigRow = modelConfig[1]

trainingXArrays = Array{Array{Float64, 2}}(industriesTotal)
trainingYArrays = Array{Array{Float64, 2}}(industriesTotal)
t=1
i=1
trainingXArrays[i] = XArrays[i][t:(t-1+trainingSize), :]
trainingYArrays[i] = YArrays[i][t:(t-1+trainingSize), :]

observations = size(trainingXArrays[1])[1]

l1, l2, l3, l4 = modelConfig[30,:]
env = Gurobi.Env()
bCols = size(XArrays[1])[2]
M = JuMP.Model(solver = GurobiSolver(env, OutputFlag = 0, Threads=(nprocs()-1)))
@variables M begin
        b[1:10, 1:bCols]
end
@objective(M,Min,l1*(sum(sqrt((trainingYArrays[1][t]-trainingXArrays[1][t,:]*b[1,:]')^2) for t=1:observations)))

solve(M)

#	println("solved by worker $(myid())")
return getvalue(b)


function runCEO(trainingXArrays, trainingYArrays, modelConfigRow)
    l1, l2, l3, l4 = modelConfigRow

    bCols = size(trainingXArrays[1])[2]
	M = JuMP.Model(solver = GurobiSolver(env, OutputFlag = 0, Threads=(nprocs()-1)))
	@variables M begin
			b[1:bCols]
			t[1:bCols]
			w
	end
	@objective(M,Min,0.5*w+gamma*ones(bCols)'*t)
	@constraint(M, soc, norm( [1-w;2*(Xtrain*b-Ytrain)] ) <= 1+w)
	@constraint(M,  b .<= t)
	@constraint(M, -t .<= b)

	solve(M)

#	println("solved by worker $(myid())")
	return getvalue(b)


function generateXandYs(industries, modelMatrix)
    industriesTotal = length(industries)
    for i = 1:industriesTotal
        industry = industries[i]
        mainData = loadIndexDataLOGReturn(industry, path)

        #path = "/zhome/9f/d/88706/SpecialeCode/"
        #path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
        dateAndRecession = Array(mainData[:,end-1:end])
        mainDataArr = Array(mainData[:,1:end-2])

        colNames = names(mainData)

        nRows = size(mainDataArr)[1]
        nCols = size(mainDataArr)[2]

        #fileName = path*"/Results/IndexData/LassoTests/240-1/2401_Shrink_"
        #fileName = path*"/Results/Test/Parallel/"
        #fileName = path*"Results/IndexData/LassoTest/"*industry*"/"*string(trainingSize)*"-1/"*string(trainingSize)*"_"

        #Reset HPC path
        #path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
        #cd(path)
        fileName = path*"Results/IndexData/CEOTest/"*industry*"/"*string(trainingSize)*"-1/"*string(trainingSize)*"_"

        mainYarr = mainDataArr[:, nCols:nCols]

        mainXarr = mainDataArr[:,1:nCols-1]

        VIX = modelMatrix[i, 1]
        raw = modelMatrix[i, 2]
        expTrans = modelMatrix[i, 3]
        timeTrans = modelMatrix[i, 4]
        TA = modelMatrix[i,5]

        if raw == 1
            fileName = fileName*"Raw"
            mainXarr = mainXarr[:,1:10]
        else
            if VIX == 1
                fileName = fileName*"VIX_Macro"
                mainXarr = mainXarr[:,1:end-1]
            else
                fileName = fileName*"Macro"
            end
        end

        # Transform with time elements
        if timeTrans == 1
            fileName = fileName*"Time"
            mainXarr = expandWithTime3612(mainXarr)
        end

        # Transform #
        if expTrans == 1
            fileName = fileName*"Exp"
            mainXarr = expandWithTransformations(mainXarr)
        end

        if TA == 1
            fileName = fileName*"TA"
            mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))
        end

        # Standardize
        standX = mainXarr#zScoreByColumn(mainXarr)
        standY = mainYarr

        XArrays[i] = standX
        YArrays[i] = standY
    end
    return XArrays, YArrays
end

"""
Keeps track of returns for both the proposed method and 1/N.
Needs the overall array as well as the two next returns to push
"""

function trackPortfolioReturns(ReturnOverall, individualReturn, newPeriodReturn, Return1NOverall, Individual1NReturn, period1NReturn)
    currentLength = length(ReturnOverall)
    ReturnOverall = push!(ReturnOverall, ReturnOverall[currentLength]*newPeriodReturn)
    individualReturn = push!(individualReturn, newPeriodReturn)

    current1NLength = length(Return1NOverall)
    Return1NOverall = push!(Return1NOverall, Return1NOverall[current1NLength]*period1NReturn)
    Individual1NReturn = push!(Individual1NReturn, period1NReturn)
end

function trackPortfolioWeights(weights, periodWeights, iteration)
    weights[iteration, 1:10] = periodWeights
end
