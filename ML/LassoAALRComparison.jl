using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

industry = "NoDur"
folder = "240"
for i = [12, 24, 36, 48, 120, 240]
    #folder = string(i)
    #writeFile()
end

# Load all 3 AALR data
nRows = 845
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/"*string(folder)*"-1/"
cd(path)

AALRMainData = zeros(3, nRows, 1472)
for i = 1:nRows
    f = open(string(i)*"AALRBestK_240.csv")
    readline(f)
    s = readline(f)
    AALRMainData[1, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    AALRMainData[2, i, :] = parse.(Float64, split(s, ","))

    s = readline(f)
    AALRMainData[3, i, :] = parse.(Float64, split(s, ","))
end
AALRMainData = AALRMainData[:,:,9:end]

# Get column names
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexDataDiff/"
industry = "NoDur"
mainData = loadIndexDataLOGReturn(industry, path)
colNames = names(mainData)
featureNames = expandColNamesTimeToString(colNames[1:end-3])
s = split(expandedColNamesToString(featureNames,true,true),",")
s = Array{String}(s)

# Load all best RMSE bsolved
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
path = path*"Results/IndexData/LassoTest/NoDur/"
summary = CSV.read(path*"Summary "*folder*".csv",nullable = false)
minIndex = summary[end,end]


path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
path = path*"Results/IndexData/LassoTest/NoDur/"*string(folder)*"-1/"
cd(path)
path
dType = [Float64]
for i = 1:1463
	dType = vcat(dType,Float64)
end

LassoMainData = CSV.read("240_VIX_MacroTimeExpTA_bMatrix240_"*string(minIndex)".csv", header=s,
    delim = ',', nullable=false, types = dType)

#Remove all columns that are never active for both AALR and LASSO
LassoMainDataActive = Array{Float64}(size(LassoMainData)[1],0)
LassoHeaderActive = Array{String}(0)
LassoParamUsage = Array{Float64}(0)
for i=1:size(LassoMainData)[2]
	paramUsage = countnz(LassoMainData[:,i])
	if paramUsage > 0
		LassoParamUsage = vcat(LassoParamUsage, paramUsage)
		LassoMainDataActive = hcat(LassoMainDataActive,LassoMainData[:,i])
		LassoHeaderActive = vcat(LassoHeaderActive,s[i])
	end
end
LassoTotalNumActiveParam = size(LassoMainDataActive)[2]
println("Number of active parameters is ", LassoTotalNumActiveParam)
LassoAvgParamUsage = mean(LassoParamUsage)/size(LassoMainData)[2]
println("Most used parameter ", maximum(LassoParamUsage)/size(LassoMainData)[1])
LassoParamUsageCopy = LassoParamUsage
test = zeros(20,2)
for i = 1:20
    maxind = findmax(LassoParamUsageCopy)[2]
    LassoParamUsageCopy[maxind] = 0
    test[i,1] = maxind
    println(i, " most used prediction term is index ", maxind)
end
println("An active parameter in LASSO is used ", mean(LassoParamUsage)/size(LassoMainData)[2]*100,"% of the time")

AALRMainData[1,:,:]
AALRMainDataActive = Array{Float64}(size(AALRMainData)[2],0)
AALRHeaderActive = Array{String}(0)
AALRParamUsage = Array{Float64}(0)
for i=1:size(AALRMainData[1,:,:])[2]
    paramUsage = length(find(x -> x>2e-6 || x<-2e-6, AALRMainData[1,:,i]))
	if paramUsage > 0
		AALRParamUsage = vcat(AALRParamUsage, paramUsage)
		AALRMainDataActive = hcat(AALRMainDataActive,AALRMainData[1,:,i])
		AALRHeaderActive = vcat(AALRHeaderActive,s[i])
	end
end
AALRTotalNumActiveParam = size(AALRMainDataActive)[2]
println("Number of active parameters is ",AALRTotalNumActiveParam)
AALRAvgParamUsage = mean(AALRParamUsage)/size(AALRMainData[1,:,:])[2]
println("Most used parameter ", maximum(AALRParamUsage)/size(LassoMainData)[1])
findmax(AALRParamUsage)
AALRParamUsageCopy = AALRParamUsage
for i = 1:20
    maxind = findmax(AALRParamUsageCopy)[2]
    AALRParamUsageCopy[maxind] = 0
    test[i,2] = maxind
    println(i, " most used prediction term is index ", maxind)
end
println("An active parameter in AALR is used ", mean(AALRParamUsage)/size(AALRMainData[1,:,:])[2]*100,"% of the time")

for i = 1:20
    println("At i = ", i, " we found ", find(x -> x == test[i,1], test[:,2]))
end

#Count average number of parameters in AALR and LASSO
LassoMainDataArr = Array{Float64}(LassoMainData)
LassoKArr = Array{Int64}(size(LassoMainData)[1])
for i=1:size(LassoMainData)[1]
	LassoKArr[i] = countnz(LassoMainDataArr[i,:])
end
#lassoAvgK = lassoAvgK/(size(LassoMainData)[1])
lassoAvgK = mean(LassoKArr)
lassoStd = std(LassoKArr)

AALRMainDataArr = Array{Float64}(AALRMainData[1,:,:])
AALRKArr = Array{Int64}(size(AALRMainDataArr)[1])
for i=1:size(AALRMainDataArr)[1]
	#AALRAvgK += countnz(AALRMainDataArr[i,:])
    AALRKArr[i] = size(find(x -> x >= 1e-6 || x <=-1e-6, AALRMainDataArr[i,:]))[1]
end
AALRAvgK = mean(AALRKArr)
AALRStd = std(AALRKArr)

#Potentially plot variable selection and compare it to recession
