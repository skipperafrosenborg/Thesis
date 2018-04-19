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
portfolioClassTotal = []
portfolioClassTotal = push!(portfolioClassTotal, 1)
portfolioClassIndividual = []
portfolioClassIndividual = push!(portfolioClassIndividual, 0)
wClasstracker = zeros((nRows-trainingObservations-1),10)


wClass = repeat([0.1], outer = 10)
portfolioClassTotal = []
portfolioClassTotal = push!(portfolioClassTotal, 1)
portfolioClassIndividual = []
portfolioClassIndividual = push!(portfolioClassIndividual, 0)

for rStart = 1:(nRows-trainingObservations-1)
    startRow = rStart
    endRow = rStart+trainingObservations
    mainXarr = mainDataArr[startRow:endRow,1:10]
    indexes = size(mainXarr)[2]

    #creating the covariance matrix of the training data
    Sigma = cov(mainXarr)

    #creating the mean of the returns for SAA (forecasts should be used instead)
    LassoOutputHere = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    indicators = LassoOutputHere
    wStar = createWeighting(indicators)

    wMVtracker[rStart,:] = wStar

    #creating the forecastRow in arithmetic returns (not log returns)
    #then we can multiply and sum across the row
    forecastRow = exp(mainDataArr[(endRow+1):(endRow+1),1:10])

    periodReturn = Array(forecastRow*wStar)[1]
    currentLength = length(portfolioClassTotal)
    portfolioClassTotal = push!(portfolioClassTotal, portfolioClassTotal[currentLength]*periodReturn)
    portfolioClassIndividual = push!(portfolioClassIndividual, periodReturn)

    period1NReturn = Array(forecastRow*w1N)[1]
    current1NLength = length(portfolio1NTotal)
    portfolio1NTotal = push!(portfolio1NTotal, portfolio1NTotal[current1NLength]*period1NReturn)
    portfolio1NIndividual = push!(portfolio1NIndividual, period1NReturn)
end

#Checking the total return of the portfolio by checking last index
portfolioClassTotal[length(portfolioClassTotal)]
portfolio1NTotal[length(portfolio1NTotal)]

portfolioClassTotal = convert(Array{Float64,1}, portfolioClassTotal)
portfolioClassIndividual = convert(Array{Float64,1}, portfolioClassIndividual)
(mean(portfolioClassIndividual)-1)*100
std(portfolioClassIndividual)
SharpeRatioMV = (mean(portfolioClassIndividual)-1)/std(portfolioClassIndividual)

portfolio1NTotal = convert(Array{Float64,1}, portfolio1NTotal)
portfolio1NIndividual = convert(Array{Float64,1}, portfolio1NIndividual)
(mean(portfolio1NIndividual)-1)*100
std(portfolio1NIndividual)
SharpeRatio1N = (mean(portfolio1NIndividual)-1)/std(portfolio1NIndividual)

combinedPortfolios = hcat(portfolioClassTotal, portfolio1NTotal, portfolioClassIndividual, portfolio1NIndividual)
writedlm("Class100obsversus1N.csv", combinedPortfolios,",")


writedlm("Class100obsPortfolioWeights.csv", wMVtracker,",")
