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
path = "$(homedir())/Documents/GitHub/Thesis/Data/IndexDataDiff"

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexData/"
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/IndexData/"

trainingSize = 240
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
w1N = repeat([0.1], outer = 11) #1/N weights
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
weightsCEO     = zeros(nRows-trainingSize, 10)

#Establishing perfect results in order to avoid doing same mean-variance calculation over and over
for t=1:(nRows-trainingSize-2)
    trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)
    valY = zeros(10)
    for i = 1:10
        valY[i] = validationY[i][1]
    end
    weightsPerfect[t, :], returnPerfectMatrix[t] = findPerfectResults(trainingXArrays, valY, valY, gamma)
end





println("Starting CEO Validation loop")
for t=1:50# (nRows-trainingSize-2)
    println("Time $t/50")
    trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY = createDataSplits(XArrays, YArrays, t, trainingSize)

    if t <= validationPeriod
        for m = 1:amountOfModels
            betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, modelConfig[m, :], gamma))
            expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

            #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
            valY = zeros(10)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            #return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, OOSRow[1][1:10], valY)
            return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
            weightsCEO[t, 1:10]     = wStar
            return1NMatrix[t, m]      = return1N
            returnCEOMatrix[t, m]     = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, m] = calculatePvalue(return1N, returnPerfect, returnCEO)
            #trackReturn(returnCEOTotal, returnCEO)
        end
    elseif t == (validationPeriod+1)
        modelMeans = mean(PMatrix, 1)
        for i = 1:bestModelAmount
            bestModelIndex = indmax(modelMeans)
            bestModelConfig[i] = modelConfig[bestModelIndex, :]
            bestModelIndexes[i] = bestModelIndex
            #place 0 at index place now and take new max as "next best model"
            modelMeans[bestModelIndex] = 0
        end

        for m = 1:bestModelAmount
            betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, bestModelConfigs[m, :]))
            expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

            #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
            valY = zeros(10)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            return1N, returnPerfect, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
            weightsCEO[t, 1:10]     = wStar
            return1NMatrix[t, bestModelIndexes[m]]      = return1N
            returnCEOMatrix[t, bestModelIndexes[m]]     = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, bestModelIndexes[m]] = calculatePvalue(return1N, returnPerfect, returnCEO)
        end
    else
        for m = 1:bestModelAmount
            betaArray, U = @time(runCEO(trainingXArrays, trainingYArrays, bestModelConfigs[m, :]))
            expectedReturns = generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)

            #Need to send OOSRow to mean-variance optimization to get "perfect information" since validationY is the values in OOSRow[1:10]
            valY = zeros(10)
            for i = 1:10
                valY[i] = validationY[i][1]
            end
            return1N, returnCEO, wStar = performMVOptimization(expectedReturns, U, gamma, valY, valY)
            weightsCEO[t, 1:10]     = wStar
            return1NMatrix[t, bestModelIndexes[m]]      = return1N
            returnCEOMatrix[t, bestModelIndexes[m]]     = returnCEO
            returnPerfect = returnPerfectMatrix[t]
            println("1N returns is $return1N, returnPerfect is $returnPerfect and returnCEO is $returnCEO")
            PMatrix[t, bestModelIndexes[m]] = calculatePvalue(return1N, returnPerfect, returnCEO)
        end

    end

end
Array(return1NMatrix[1:200, 1])
returnPerfectMatrix[1:200, 1]
returnCEOMatrix[1:200, 1]
combinedPortfolios = hcat(returnPerfectMatrix[1:200, 1], return1NMatrix[1:200, 1], returnCEOMatrix[1:200, 1], PMatrix[1:200, 1])
writedlm("returnPvalueOutcome1to200.csv", combinedPortfolios, ",")






















function createDataSplits(XArrays, YArrays, t, trainingSize)
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
    return trainingXArrays, trainingYArrays, validationXRows, validationY, OOSXArrays, OOSYArrays, OOSRow, OOSY
end


function calculatePvalue(return1N, returnPerfect, returnCEO)
    PValue = 1 - (returnCEO-returnPerfect)/(return1N-returnPerfect)
    return PValue
end

function findPerfectResults(trainingXArrays, Xrow, Yvalues, gamma)
    indexes = 10
    Sigma =  cov(trainingXArrays[1][:,1:10])

    #A=U^(T)U where U is upper triangular with real positive diagonal entries
    F = lufact(Sigma)
    U = F[:U]  #Cholesky factorization of Sigma

    M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
    @variables M begin
            w[1:indexes]
            u[1:indexes]
            z
            y
    end

    @objective(M,Min, gamma*y - Yvalues'*w)
    @constraint(M, 0 .<= w)
    @constraint(M, sum(w[i] for i=1:indexes) == 1)
    @constraint(M, norm([2*U'*w;y-1]) <= y+1)
    solve(M)
    wPerfect = getvalue(w)
    forecastRow = (exp10(Xrow')-1)*100
    periodPerfectReturn = forecastRow*wPerfect

    return wPerfect, periodPerfectReturn
end

function performMVOptimization(expectedReturns, U, gamma, Xrow, Yvalues)
    indexes = 10

    M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
    @variables M begin
            w[1:indexes]
            u[1:indexes]
            z
            y
    end

    @objective(M,Min, gamma*y - expectedReturns'*w)
    @constraint(M, 0 .<= w)
    @constraint(M, sum(w[i] for i=1:indexes) == 1)
    @constraint(M, norm([2*U'*w;y-1]) <= y+1)
    solve(M)
    wStar = getvalue(w)

    forecastRow = (exp(Xrow)-1)*100

    periodReturn = forecastRow'*wStar
    period1NReturn = forecastRow'*w1N

    return period1NReturn, periodReturn, wStar
end



function generateExpectedReturns(betaArray, trainingXArrays, trainingYArrays, validationXRows)
    industriesTotal = 10

    expectedReturns = zeros(10)
    bRows = 240
    bCols = floor(Int,zeros(industriesTotal))
    for i = 1:industriesTotal
        bCols[i] = size(trainingXArrays[i])[2]
    end

    for i = 1:10
        errorArray = (trainingXArrays[i]*betaArray[i,1:bCols[i]]-trainingYArrays[i])*(1/bRows)
        errorSum = sum(errorArray)
        expectedReturns[i] = validationXRows[i][:]'*betaArray[i,1:bCols[i]]
    end
    return expectedReturns
end






function runCEO(trainingXArrays, trainingYArrays, modelConfigRow, gamma)
    industriesTotal = 10
    periodMean = zeros(10)
    for i = 1:10
        periodMean[i] = mean(trainingXArrays[1][:,i])
    end
    l1, l2, l3, l4 = modelConfigRow
    env = Gurobi.Env()

    maxPredictors = 1412
    bCols = floor(Int,zeros(industriesTotal))
    Sigma =  cov(trainingXArrays[1][:,1:10])

    #A=U^(T)U where U is upper triangular with real positive diagonal entries
    F = lufact(Sigma)
    U = F[:U]  #Cholesky factorization of Sigma

    for i = 1:industriesTotal
        bCols[i] = size(trainingXArrays[i])[2]
    end
    gamma = gamma
    bRows = size(trainingXArrays[1])[1]
    M = JuMP.Model(solver = GurobiSolver(env, OutputFlag = 0, Threads=(nprocs()-1)))

    @variables M begin
            b[1:industriesTotal, 1:maxPredictors] #Beta in LASSO; industry i and predictors 1:1304
            z[1:industriesTotal, 1:maxPredictors] #auxilliary for 1norm
            q[1:industriesTotal] #auxilliary for norm
            f[1:industriesTotal, 1:bRows] #expected return for industry i at time t
            w[1:industriesTotal, 1:bRows] #weights for industry i at time t
            y[1:bRows] #auxilliary variables for MV optimization
    end

    #@objective(M,Min,sum(0.5*q[i] + l2*ones(maxPredictors)'*z[i,:] for i=1:industriesTotal)+ l3*(sum(gamma*y[t]-f[:,t]'*w[:,t] for t=1:bRows)) - l4*(sum(w[i,t]*trainingYArrays[i][t] for i=1:10, t=1:bRows)))

    @objective(M,Min,sum(l1*q[i] + l2*ones(maxPredictors)'*z[i,:] for i=1:industriesTotal)+ l3*(sum(gamma*y[t]-sum(f[i,t] for i=1:10) for t=1:bRows)) - l4*(sum(w[i,t]*trainingYArrays[i][t] for i=1:10, t=1:bRows)))

    for i = 1:industriesTotal
        for t = 1:bRows
            @constraint(M, (periodMean[i]+1000)*w[i,t] >= f[i,t])
        end
    end
    for i = 1:industriesTotal
        @constraint(M, norm( [1-q[i];2*(trainingXArrays[i]*b[i,1:bCols[i]]-trainingYArrays[i])] ) <= 1+q[i]) #second order cone constraint SOC
    end
    @constraint(M,  b .<= z)
    @constraint(M, -z .<= b)


    for i = 1:10
        errorArray = (trainingXArrays[i]*b[i,1:bCols[i]]-trainingYArrays[i])*(1/bRows)
        errorSum = sum(errorArray)
        for t = 1:bRows
            prediction = trainingXArrays[i][t, :]'*b[i,1:bCols[i]]
            @constraint(M, prediction + errorSum >= f[i, t]) ##== originally
        end
    end

    @constraint(M, 0 .<= w)
    for t = 1:bRows
        @constraint(M, sum(w[i,t] for i=1:industriesTotal) == 1)
        @constraint(M, norm([2*U'*w[:,t];y[t]-1]) <= y[t]+1)
    end

    @time(solve(M))
    betaArray = getvalue(b)
    #Insert shrinking here
    return betaArray, U
end


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
        standX = zScoreByColumn(mainXarr)
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
