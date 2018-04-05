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
path = "$(homedir())/Documents/GitHub/Thesis/Data/MonthlyReturns"
#mainData = loadIndexDataNoDurLOGReturn(path)

mainData = loadIndexDataNoDur(path)
mainData = loadIndexDataNoDurLOGReturn(path)

mainDataArr = Array(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

startRow = 1
endRow = 100
mainXarr = mainDataArr[startRow:endRow,1:10]
indexes = size(mainXarr)[2]

#creating the covariance matrix of the training data
Sigma = cov(mainXarr)

#creating the mean of the returns (here our forecasts should be used instead)
Mu = mean(mainXarr, 1)'

#setting risk aversion.
gamma = 1

portfolioReturnTotal = []
portfolioReturnTotal = push!(portfolioReturnTotal, 1)

M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
@variables M begin
        w[1:indexes]
end
@objective(M,Min,w'*Sigma*w)
@constraint(M, 0 .<= w)
@constraint(M, sum(w[i] for i=1:indexes) == 1)

solve(M)
wStar = getvalue(w)

forecastRow = mainDataArr[(endRow+1):(endRow+1),1:10]
periodReturn = Array((endRow+1)*wStar)[1]
currentLength = length(portfolioReturnTotal)
portfolioReturnTotal = push!(portfolioReturnTotal, portfolioReturnTotal[currentLength]+periodReturn)


##LOOP TO TEST IT
portfolioMVTotal = []
portfolioMVTotal = push!(portfolioMVTotal, 1)
portfolioMVIndividual = []
portfolioMVIndividual = push!(portfolioMVIndividual, 0)

w1N = repeat([0.1], outer = 10)
portfolio1NTotal = []
portfolio1NTotal = push!(portfolio1NTotal, 1)
portfolio1NIndividual = []
portfolio1NIndividual = push!(portfolio1NIndividual, 0)

for rStart = 1:985

    startRow = rStart
    endRow = rStart+100
    mainXarr = mainDataArr[startRow:endRow,1:10]
    indexes = size(mainXarr)[2]

    #creating the covariance matrix of the training data
    Sigma = cov(mainXarr)

    #creating the mean of the returns (here our forecasts should be used instead)
    Mu = mean(mainXarr, 1)'

    #setting risk aversion.
    gamma = 1

    riskFreeRate = 0
    M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
    @variables M begin
            w[1:indexes]
    end
    @objective(M,Min,0.5*w'*Sigma*w)
    @constraint(M, 0 .<= w)
    @constraint(M, sum(w[i] for i=1:indexes) == 1) #tangency portfolio since the weights of risky assets sum to 1

    solve(M)
    wStar = getvalue(w)

    forecastRow = mainDataArr[(endRow+1):(endRow+1),1:10]
    periodReturn = Array(forecastRow*wStar)[1]
    currentLength = length(portfolioMVTotal)
    portfolioMVTotal = push!(portfolioMVTotal, portfolioMVTotal[currentLength]*(1+periodReturn/100))
    portfolioMVIndividual = push!(portfolioMVIndividual, periodReturn)

    period1NReturn = Array(forecastRow*w1N)[1]
    current1NLength = length(portfolio1NTotal)
    portfolio1NTotal = push!(portfolio1NTotal, portfolio1NTotal[current1NLength]*(1+period1NReturn/100))
    portfolio1NIndividual = push!(portfolio1NIndividual, period1NReturn)
end

portfolioMVTotal[length(portfolioMVTotal)]
portfolio1NTotal[length(portfolio1NTotal)]

portfolioMVTotal
portfolio1NTotal
combinedPortfolios = hcat(portfolioMVTotal, portfolio1NTotal, portfolioMVIndividual, portfolio1NIndividual)
writedlm("MVversus1N.csv", combinedPortfolios,",")







#### WITH RISK AVERSION
mainData = loadIndexDataNoDur(path)
mainData = loadIndexDataNoDurLOGReturn(path)

mainDataArr = Array(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

startRow = 1
endRow = 100
mainXarr = mainDataArr[startRow:endRow,1:10]
indexes = size(mainXarr)[2]

#creating the covariance matrix of the training data
Sigma = cov(mainXarr)

#creating the mean of the returns (here our forecasts should be used instead)
Mu = mean(mainXarr, 1)'

#setting risk aversion.
gamma = 1

portfolioReturnTotal = []
portfolioReturnTotal = push!(portfolioReturnTotal, 1)


M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
@variables M begin
        w[1:indexes]
end
@objective(M,Min, w'*Sigma*w-(gamma*Mu'*w)[1])
@constraint(M, 0 .<= w)
@constraint(M, sum(w[i] for i=1:indexes) == 1)

solve(M)
wStar = getvalue(w)

forecastRow = mainDataArr[(endRow+1):(endRow+1),1:10]
periodReturn = Array(forecastRow*wStar)[1]
currentLength = length(portfolioReturnTotal)
portfolioReturnTotal = push!(portfolioReturnTotal, portfolioReturnTotal[currentLength]+periodReturn)

##LOOP TO TEST IT
portfolioMVTotal = []
portfolioMVTotal = push!(portfolioMVTotal, 1)
portfolioMVIndividual = []
portfolioMVIndividual = push!(portfolioMVIndividual, 0)

w1N = repeat([0.1], outer = 10)
portfolio1NTotal = []
portfolio1NTotal = push!(portfolio1NTotal, 1)
portfolio1NIndividual = []
portfolio1NIndividual = push!(portfolio1NIndividual, 0)

for rStart = 1:985

    startRow = rStart
    endRow = rStart+100
    mainXarr = mainDataArr[startRow:endRow,1:10]
    indexes = size(mainXarr)[2]

    #creating the covariance matrix of the training data
    Sigma = cov(mainXarr)

    #creating the mean of the returns (here our forecasts should be used instead)
    Mu = mean(mainXarr, 1)'

    #setting risk aversion.
    gamma = 1

    M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
    @variables M begin
            w[1:indexes]
    end
    @objective(M,Min, w'*Sigma*w-(gamma*Mu'*w)[1])
    @constraint(M, 0 .<= w)
    @constraint(M, sum(w[i] for i=1:indexes) == 1)

    solve(M)
    wStar = getvalue(w)

    forecastRow = mainDataArr[(endRow+1):(endRow+1),1:10]
    periodReturn = Array(forecastRow*wStar)[1]
    currentLength = length(portfolioMVTotal)
    portfolioMVTotal = push!(portfolioMVTotal, portfolioMVTotal[currentLength]+periodReturn)
    portfolioMVIndividual = push!(portfolioMVIndividual, periodReturn)

    period1NReturn = Array(forecastRow*w1N)[1]
    current1NLength = length(portfolio1NTotal)
    portfolio1NTotal = push!(portfolio1NTotal, portfolio1NTotal[current1NLength]+period1NReturn)
    portfolio1NIndividual = push!(portfolio1NIndividual, period1NReturn)
end

portfolioMVTotal[length(portfolioMVTotal)]
portfolio1NTotal[length(portfolio1NTotal)]

portfolioMVTotal
portfolio1NTotal
combinedPortfolios = hcat(portfolioMVTotal, portfolio1NTotal, portfolioMVIndividual, portfolio1NIndividual)
writedlm("MVversus1N.csv", combinedPortfolios,",")
