#Code written by Skipper af Rosenborg and Esben Bager
using StatsBase
using DataFrames
using CSV
using JuMP
include("SupportFunction.jl")

#Set path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/Diabetes64/"
cd(path)

#Set amount of columns
nCols = 262

#Set dataname
fName = "Diabetes64"

totalLog = zeros(30,nCols)

#Load AALR data, calculate max coorelation and output to file
for i = 1:10
    println(i)
    mainData = Array(CSV.read(string(i)*"Xtrain.csv", delim = ',', nullable=false))
    best3Beta = CSV.read(string(i)*"_"*fName*".csv", delim = ',', nullable=false, types = fill(Float64,nCols), header=false, datarow=2)

    best3BetaArr = Array(best3Beta)

    best3Beta[1,1] = findMaxCor(mainData, best3BetaArr[1,7:end])
    best3Beta[2,1] = findMaxCor(mainData, best3BetaArr[2,7:end])
    best3Beta[3,1] = findMaxCor(mainData, best3BetaArr[3,7:end])

    CSV.write(string(i)*fName*".csv", best3Beta)

    totalLog[1+(i-1)*3:3+(i-1)*3,:]= Array(best3Beta)
end
writedlm("TotalLog.csv", totalLog, ",")

#Initilise LASSO arrays
lassoBestK = zeros(nCols,nCols)
lassoSummary = zeros(nCols,nCols)
totalLog = zeros(30,nCols-1)

dataT = [Float64]
for j = 1:nCols-2
    dataT = vcat(dataT, Float64)
end

for i = 1:10
    #Load data
    println(i)
    mainData = Array(CSV.read(""*string(i)*"Xtrain.csv", delim = ',', nullable=false))
    dataInput = CSV.read(string(i)*"_Lasso"*fName*".csv", delim = ',', nullable=false, types = dataT, datarow=2, header=false)

    dataInputArr = Array{Float64}(dataInput)
    #Check if any lambda regulations results in j active prediciton terms
    for j = 1:nCols
        lassoBestK[j,1] = j
        lassoSummary[j,1] = j
        currentK = find(k -> (k==j), dataInputArr[:,4])
        if isempty(currentK)
            continue
        end

        #Find max calue within the places we have j active prediction terms
        val, index = findmax(dataInputArr[currentK,2])

        #Summary to be average over all datasets
        lassoSummary[j,2:end] += dataInputArr[currentK[index],:]

        #summary for current dataset
        lassoBestK[j,2:end] = dataInputArr[currentK[index],:]
        lassoBestK[j,2] = findMaxCor(mainData, lassoBestK[j,7:end])
    end

    #Find 3 highest R^2 values
    for j = 1:3
        val, index = findmax(dataInputArr[:,3])
        totalLog[j+(i-1)*3:j+(i-1)*3,:]= Array(dataInputArr[index, :])
        dataInputArr[index, 3] = 0
    end

    writedlm(string(i)*"LassoSummary.csv", lassoBestK,",")
end

writedlm("TotalLasso.csv", totalLog, ",")

lassoSummary[:,2:end] = lassoSummary[:,2:end]/10
writedlm("TotalLassoSummary.csv", lassoSummary,",")
