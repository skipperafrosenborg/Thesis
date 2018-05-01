using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

industry = "NoDur"
folder = "12"
for i = [12, 24, 36, 48, 120, 240]
    folder = string(i)
    writeFile()
end

# Load all 3 AALR data
nRows = 1070
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Results/IndexData/AALRTest/"*string(folder)*"-1/"
cd(path)

AALRMainData = zeros(3, nRows, 1472)
for i = 1:nRows
    f = open(string(i)*"AALRBestK.csv")
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
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/IndexData/"
industry = "NoDur"
mainData = loadIndexDataLOGReturn(industry, path)
colNames = names(mainData)
featureNames = expandColNamesTimeToString(colNames[1:end-3])
s = split(expandedColNamesToString(featureNames,true),",")
s = Array{String}(s)

# Load all best RMSE bsolved
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
path = path*"Results/IndexData/LassoTest/NoDur/ParametersSelected/"*string(folder)*"-1/"
cd(path)
path
dType = [Float64]
for i = 1:1463
	dType = vcat(dType,Float64)
end

LassoMainData = CSV.read("bSolved.csv", header=s,
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
LassoAvgParamUsage = mean(LassoParamUsage)/size(LassoMainData)[2]
println("An active parameter is used ", mean(LassoParamUsage)/size(LassoMainData)[2]*100,"% of the time")

AALRMainData[1,:,:]
AALRMainDataActive = Array{Float64}(size(AALRMainData)[2],0)
AALRHeaderActive = Array{String}(0)
AALRParamUsage = Array{Float64}(0)
for i=1:size(AALRMainData[1,:,:])[2]
	paramUsage = countnz(AALRMainData[1,:,i])
	if paramUsage > 0
		AALRParamUsage = vcat(AALRParamUsage, paramUsage)
		AALRMainDataActive = hcat(AALRMainDataActive,AALRMainData[1,:,i])
		AALRHeaderActive = vcat(AALRHeaderActive,s[i])
	end
end
AALRTotalNumActiveParam = size(AALRMainDataActive)[2]
AALRAvgParamUsage = mean(AALRParamUsage)/size(AALRMainData[1,:,:])[2]
println("An active parameter is used ", mean(AALRParamUsage)/size(AALRMainData[1,:,:])[2]*100,"% of the time")

#Count average number of parameters in AALR and LASSO
lassoAvgK = 0
LassoMainDataArr = Array{Float64}(LassoMainData)
for i=1:size(LassoMainData)[1]
	lassoAvgK += countnz(LassoMainDataArr[i,:])
end
lassoAvgK = lassoAvgK/(size(LassoMainData)[1])

AALRAvgK = 0
AALRMainDataArr = Array{Float64}(AALRMainData[1,:,:])
for i=1:size(AALRMainDataArr)[1]
	AALRAvgK += countnz(AALRMainDataArr[i,:])
end
AALRAvgK = AALRAvgK/(size(AALRMainDataArr)[1])

#Potentially plot variable selection and compare it to recession
