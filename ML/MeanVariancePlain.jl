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

mainData = loadIndexDataNoDurLOGReturn(path)

mainDataArr = Array(mainData)

startRow = 1
endRow = 100
trainingObservations = 100
mainXarr = mainDataArr[startRow:endRow,1:10]
indexes = size(mainXarr)[2]


#### WITH RISK AVERSION

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

##LOOP TO TEST IT
portfolioMVTotal = []
portfolioMVTotal = push!(portfolioMVTotal, 1)
portfolioMVIndividual = []
portfolioMVIndividual = push!(portfolioMVIndividual, 0)
wMVtracker = zeros((nRows-trainingObservations-1),10)


w1N = repeat([0.1], outer = 10)
portfolio1NTotal = []
portfolio1NTotal = push!(portfolio1NTotal, 1)
portfolio1NIndividual = []
portfolio1NIndividual = push!(portfolio1NIndividual, 0)

for rStart = 1:(nRows-trainingObservations-1)
    startRow = rStart
    endRow = rStart+trainingObservations
    mainXarr = mainDataArr[startRow:endRow,1:10]
    indexes = size(mainXarr)[2]

    #creating the covariance matrix of the training data
    Sigma = cov(mainXarr)

    #creating the mean of the returns for SAA (forecasts should be used instead)
    Mu = mean(mainXarr, 1)'

    #setting risk aversion.
    gamma = 10

    M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
    @variables M begin
            w[1:indexes]
    end
    @objective(M,Min, gamma*w'*Sigma*w-(Mu'*w)[1]) #efficient frontier
    @constraint(M, 0 .<= w)
    @constraint(M, sum(w[i] for i=1:indexes) == 1)

    solve(M)
    wStar = getvalue(w)
    for i = 1:length(wStar)
        if wStar[i] >=0.5
            println("Index $i is chosen")
        end
    end
    wMVtracker[rStart,:] = wStar

    #creating the forecastRow in arithmetic returns (not log returns)
    #then we can multiply and sum across the row
    forecastRow = exp(mainDataArr[(endRow+1):(endRow+1),1:10])

    periodReturn = Array(forecastRow*wStar)[1]
    currentLength = length(portfolioMVTotal)
    portfolioMVTotal = push!(portfolioMVTotal, portfolioMVTotal[currentLength]*periodReturn)
    portfolioMVIndividual = push!(portfolioMVIndividual, periodReturn)

    period1NReturn = Array(forecastRow*w1N)[1]
    current1NLength = length(portfolio1NTotal)
    portfolio1NTotal = push!(portfolio1NTotal, portfolio1NTotal[current1NLength]*period1NReturn)
    portfolio1NIndividual = push!(portfolio1NIndividual, period1NReturn)
end

#Checking the total return of the portfolio by checking last index
portfolioMVTotal[length(portfolioMVTotal)]
portfolio1NTotal[length(portfolio1NTotal)]

portfolioMVTotal = convert(Array{Float64,1}, portfolioMVTotal)
portfolioMVIndividual = convert(Array{Float64,1}, portfolioMVIndividual)
(mean(portfolioMVIndividual)-1)*100
std(portfolioMVIndividual)
SharpeRatioMV = (mean(portfolioMVIndividual)-1)/std(portfolioMVIndividual)

portfolio1NTotal = convert(Array{Float64,1}, portfolio1NTotal)
portfolio1NIndividual = convert(Array{Float64,1}, portfolio1NIndividual)
(mean(portfolio1NIndividual)-1)*100
std(portfolio1NIndividual)
SharpeRatio1N = (mean(portfolio1NIndividual)-1)/std(portfolio1NIndividual)

combinedPortfolios = hcat(portfolioMVTotal, portfolio1NTotal, portfolioMVIndividual, portfolio1NIndividual)
writedlm("MV100obsG10versus1N.csv", combinedPortfolios,",")


writedlm("MV100obsPortfolioWeightsG10.csv", wMVtracker,",")

function meanVariancePPDOptimization(fullX, startRow, endRow, errors, meanForecasts)

end

"""
Utilizing LASSO to create a classification; 1, it goes up, 0, it goes down
The portfolio is then a 1/K, where K is the amount of indexes predicted to go up
No weighting between them or variance optimization, hence quite basic
"""
function ClassificationPortfolio(fullX, firstRow, indicators)

end
