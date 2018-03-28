using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"


"""
LASSO WITH WEIGHTED ERRORS
"""
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \n solving Convex \n \n \n")

function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma, errorWeights)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver())
	@variables M begin
	        b[1:bCols]
	        t[1:bCols]
	        w
	end
	@objective(M,Min,0.5*w+gamma*ones(bCols)'*t)
	@constraint(M, soc, norm( [1-w;2*(errorWeights.*(Xtrain*b-Ytrain))] ) <= 1+w)
	@constraint(M,  b .<= t)
	@constraint(M, -t .<= b)

	solve(M)
	bSolved = getvalue(b)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

	#OOS R-squared value
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosErrorsMean = Ypred - mean(Ypred)
	oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
	OOSRsquared = 1-(oosErrorTotal/oosErrorsMeanTotal)

	return ISRsquared, OOSRsquared
end
trainingSize = 10
errorWeights = linspace(1,0, trainingSize)
testRuns = Int64(floor(1000/trainingSize))
nGammas = 500
predictions = 5
ISR = zeros(nGammas, testRuns)
OOSR = zeros(nGammas, testRuns)
gammaArray = logspace(0, 7, nGammas)
for r = 1:testRuns
	Xtrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), 1:bCols]
	Ytrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), bCols+1]
	Xpred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISR[g, r], OOSR[g, r] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma, errorWeights)
	end
end

combinedArray = hcat(round.(gammaArray,3), ISR)
runCounter = collect(0:testRuns)
ISRcomb = vcat(runCounter', combinedArray)
writedlm("ISRsquaredweighted105.CSV", ISRcomb,",")
combinedArray = hcat(round.(gammaArray,3), OOSR)
OOSRcomb = vcat(runCounter', combinedArray)
writedlm("OOSRsquaredweighted105.CSV", OOSRcomb,",")

"""
LASSO
"""
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

mainData = loadIndexDataNoDur(path)
<<<<<<< HEAD
=======
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
>>>>>>> 4fe6210679f2fdd18244c00e7397df164bf0740d
=======
>>>>>>> parent of be6f44e... Merge branch 'master' of https://github.com/skipperafrosenborg/Thesis
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]
mainYarr = mainDataArr[:, nCols:nCols]

# Transform with time elements
mainXarr = expandWithTime3612(mainXarr)

# Transform #
mainXarr = expandWithTransformations(mainXarr)

mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))

# Standardize
standX = zScoreByColumn(mainXarr)
standY = mainDataArr[:, nCols:nCols]
#standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \n solving Convex \n \n \n")

function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver(OutputFlag = 0))
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
	bSolved = getvalue(b)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)
	#=
	#OOS R-squared value (for more predictions)
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosErrorsMean = Ypred - mean(Ypred)
	oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
	OOSRsquared = 1-(oosErrorTotal/oosErrorsMeanTotal)
	=#

	#OOS R-squared value (for single prediction)
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosSize = sum(Ypred[i]^2 for i=1:length(Ypred))
	OOSRsquared = sqrt(oosErrorTotal) #/ sqrt(oosSize)

	#Indicator Results
	YpredValue = Ypred[1]
	Yestimate = Xpred*bSolved
	YestimateValue = Yestimate[1]
	if YpredValue >= 0 && YestimateValue >= 0
		Indicator = 1
	elseif YpredValue < 0 && YestimateValue < 0
		Indicator = 1
	else
		Indicator = 0
	end

	if YestimateValue >= 0
		Indicator2 = 1
	elseif YestimateValue < 0
		Indicator2 = 0
	else
		Indicator2 = 0
	end

	return ISRsquared, OOSRsquared, Indicator, Indicator2
end



<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> parent of be6f44e... Merge branch 'master' of https://github.com/skipperafrosenborg/Thesis
nGammas = 10
trainingSize = 12
predictions = 1
testRuns = nRows-trainingSize-predictions
ISR = zeros(testRuns, nGammas)
OOSR = zeros(testRuns, nGammas)
Indi = zeros(testRuns, nGammas)
gammaArray = logspace(0, 3, nGammas)
<<<<<<< HEAD
=======
nGammas = 5
trainingSize = 48
=======
=======
>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
nGammas = 2
trainingSize = 12
>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
predictions = 1
testRuns = nRows-trainingSize-predictions
ISR = zeros(nGammas, testRuns)
OOSR = zeros(nGammas, testRuns)
Indi = zeros(nGammas, testRuns)
Indi2 = zeros(nGammas, testRuns)
gammaArray = logspace(0, 1, nGammas)
<<<<<<< HEAD
>>>>>>> 4fe6210679f2fdd18244c00e7397df164bf0740d
=======
>>>>>>> parent of be6f44e... Merge branch 'master' of https://github.com/skipperafrosenborg/Thesis
=======
>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a

for r = 1:(nRows-trainingSize-predictions)
	Xtrain = allData[r:(trainingSize+(r-1)), 1:bCols]
	Ytrain = allData[r:(trainingSize+(r-1)), bCols+1]
	Xpred  = allData[(trainingSize+(r-1)+1):(trainingSize+(r-1)+predictions), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)+1):(trainingSize+(r-1)+predictions), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
<<<<<<< HEAD
<<<<<<< HEAD
		ISR[r, g], OOSR[r, g], Indi[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
=======
		ISR[g, r], OOSR[g, r], Indi[g, r], Indi2[g, r] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
=======
		ISR[g, r], OOSR[g, r], Indi[g, r], Indi2[g, r] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
	end
	println("Row $r/",(nRows-trainingSize-predictions))
end

#=
for r = 1:testRuns
	Xtrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), 1:bCols]
	Ytrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), bCols+1]
	Xpred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISR[g, r], OOSR[g, r] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	end
end
=#
fileName = path*"/Results/IndexData/LassoTests/121TimeExpTA"
combinedArray = vcat((round.(gammaArray,3))', ISR)

runCounter = collect(0:testRuns)
ISRcomb = hcat(runCounter, combinedArray)
writedlm(fileName*"ISRsquared.CSV", ISRcomb,",")
combinedArray = vcat((round.(gammaArray,3))', OOSR)
OOSRcomb = hcat(runCounter, combinedArray)
writedlm(fileName*"OSSRsquared.CSV", OOSRcomb,",")

combinedArray = vcat((round.(gammaArray,3))', Indi)
Indicomb = hcat(runCounter, combinedArray)
writedlm(fileName*"Indi.CSV", Indicomb,",")
<<<<<<< HEAD
=======
combinedArray = hcat(round.(gammaArray,3), ISR)
runCounter = collect(0:testRuns)'
ISRcomb = vcat(runCounter, combinedArray)
writedlm("ISRTestNoDur121IndiPortfolio.CSV", ISRcomb,",")
combinedArray = hcat(round.(gammaArray,3), OOSR)
OOSRcomb = vcat(runCounter, combinedArray)
writedlm("OOSRTestNoDur121Portfolio.CSV", OOSRcomb,",")

combinedArray = hcat(round.(gammaArray,3), Indi)
Indicomb = vcat(runCounter, combinedArray)
<<<<<<< HEAD
<<<<<<< HEAD
writedlm("Indicator481TimeTAOtherNotStandY.CSV", Indicomb,",")
>>>>>>> 4fe6210679f2fdd18244c00e7397df164bf0740d
=======
writedlm("IndicatorTestNoDur121IndiPortfolio.CSV", Indicomb,",")

>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
=======
>>>>>>> parent of be6f44e... Merge branch 'master' of https://github.com/skipperafrosenborg/Thesis

=======
writedlm("IndicatorTestNoDur121IndiPortfolio.CSV", Indicomb,",")


>>>>>>> bee04ad7b02cee3642894bc88ca8f415908cae9a
combinedArray = hcat(round.(gammaArray,3), Indi2)
Indicomb = vcat(runCounter, combinedArray)
writedlm("Indicator2TestNoDur121IndiPortfolio.CSV", Indicomb,",")

"""
LASSO with daily returns
"""
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data/DailyReturns"

"""
REMEMBER: Select index to predict. Then change dataloading. Then decide
observations and predictions to include - change fileName accordingly.
"""

mainData = loadIndexDataDailyOther(path)
fileName = path*"/Results/IndexData/DailyReturnsXobservationsYpredictions"
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \n solving Convex \n \n \n")

function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver())
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
	bSolved = getvalue(b)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	isValue = sum(Ytrain[i]^2 for i=1:length(Ytrain))
	isErrorPercentage = (errorTotal / isValue) * 100
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

	#OOS R-squared value
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosValue = sum(Ypred[i]^2 for i=1:length(Ypred))
	oosErrorPercentage = (oosErrorTotal / oosValue)*100
	oosErrorsMean = Ypred - mean(Ypred)
	oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
	OOSRsquared = 1-(oosErrorTotal/oosErrorsMeanTotal)

	return isErrorPercentage, ISRsquared, oosErrorPercentage, OOSRsquared
end


testRuns = 300
nGammas = 50
trainingSize = 50
predictions = 5
ISR = zeros(testRuns, nGammas)
ISP = zeros(testRuns, nGammas)
OOSR = zeros(testRuns, nGammas)
OOSP = zeros(testRuns, nGammas)
varianceTrainArray = zeros(testRuns)
gammaArray = logspace(0, 2, nGammas)
for r = 1:testRuns
	Xtrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), 1:bCols]
	Ytrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), bCols+1]
	varianceTrainArray[r] = var(Ytrain)
	Xpred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISP[r, g], ISR[r, g], OOSP[r, g], OOSR[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	end
end

runCounter = collect(0:testRuns)
varArr = vcat("variance",varianceTrainArray)
ISR
gammaArray'
combinedArray = vcat(round.(gammaArray',3), ISR)
ISRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISRsquaredDailyOther505.CSV", ISRcomb,",")

combinedArray = vcat(round.(gammaArray',3), ISP)
ISPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISPDailyOther505.CSV", ISPcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSR)
OOSRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSRsquaredDailyOther505.CSV", OOSRcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSP)
OOSPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSPDailyOther505.CSV", OOSPcomb,",")


"""
LASSO with daily returns and weighted
"""
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data/DailyReturns"

"""
REMEMBER: Select index to predict. Then change dataloading. Then decide
observations and predictions to include - change fileName accordingly.
"""

mainData = loadIndexDataDailyOther(path)
fileName = path*"/Results/IndexData/DailyReturnsXobservationsYpredictions"
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \n solving Convex \n \n \n")

function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma, errorWeights)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver())
	@variables M begin
	        b[1:bCols]
	        t[1:bCols]
	        w
	end
	@objective(M,Min,0.5*w+gamma*ones(bCols)'*t)
	@constraint(M, soc, norm( [1-w;2*(errorWeights.*(Xtrain*b-Ytrain))] ) <= 1+w)
	@constraint(M,  b .<= t)
	@constraint(M, -t .<= b)

	solve(M)
	bSolved = getvalue(b)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	isValue = sum(Ytrain[i]^2 for i=1:length(Ytrain))
	isErrorPercentage = (errorTotal / isValue) * 100
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

	#OOS R-squared value
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosValue = sum(Ypred[i]^2 for i=1:length(Ypred))
	oosErrorPercentage = (oosErrorTotal / oosValue)*100
	oosErrorsMean = Ypred - mean(Ypred)
	oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
	OOSRsquared = 1-(oosErrorTotal/oosErrorsMeanTotal)

	return isErrorPercentage, ISRsquared, oosErrorPercentage, OOSRsquared
end


testRuns = 300
nGammas = 50
trainingSize = 50
errorWeights = linspace(0, 1, trainingSize)
predictions = 5
ISR = zeros(testRuns, nGammas)
ISP = zeros(testRuns, nGammas)
OOSR = zeros(testRuns, nGammas)
OOSP = zeros(testRuns, nGammas)
varianceTrainArray = zeros(testRuns)
gammaArray = logspace(0, 2, nGammas)
for r = 1:testRuns
	Xtrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), 1:bCols]
	Ytrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), bCols+1]
	varianceTrainArray[r] = var(Ytrain)
	Xpred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISP[r, g], ISR[r, g], OOSP[r, g], OOSR[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma, errorWeights)
	end
end

runCounter = collect(0:testRuns)
varArr = vcat("variance",varianceTrainArray)
ISR
gammaArray'
combinedArray = vcat(round.(gammaArray',3), ISR)
ISRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISRsquaredDailyOtherWeighted2505.CSV", ISRcomb,",")

combinedArray = vcat(round.(gammaArray',3), ISP)
ISPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISPDailyOtherWeighted2505.CSV", ISPcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSR)
OOSRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSRsquaredDailyOtherWeighted2505.CSV", OOSRcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSP)
OOSPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSPDailyOtherWeighted2505.CSV", OOSPcomb,",")


"""
LASSO with monthly returns
"""
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"

mainData =loadIndexDataNoDur(path)
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
standY = zScoreByColumn(mainDataArr[:, nCols:nCols])
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \n solving Convex \n \n \n")

function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver())
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
	bSolved = getvalue(b)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	isValue = sum(Ytrain[i]^2 for i=1:length(Ytrain))
	isErrorPercentage = (errorTotal / isValue) * 100
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

	#OOS R-squared value
	oosErrors = Ypred - Xpred*bSolved
	oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
	oosValue = sum(Ypred[i]^2 for i=1:length(Ypred))
	oosErrorPercentage = (oosErrorTotal / oosValue)*100
	oosErrorsMean = Ypred - mean(Ypred)
	oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
	OOSRsquared = 1-(oosErrorTotal/oosErrorsMeanTotal)

	return isErrorPercentage, ISRsquared, oosErrorPercentage, OOSRsquared
end


testRuns = 100
nGammas = 50
trainingSize = 10

predictions = 5
ISR = zeros(testRuns, nGammas)
ISP = zeros(testRuns, nGammas)
OOSR = zeros(testRuns, nGammas)
OOSP = zeros(testRuns, nGammas)
varianceTrainArray = zeros(testRuns)
gammaArray = logspace(0, 2, nGammas)
for r = 1:testRuns
	Xtrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), 1:bCols]
	Ytrain = allData[(1+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize), bCols+1]
	varianceTrainArray[r] = var(Ytrain)
	Xpred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)*trainingSize):(trainingSize+(r-1)*trainingSize+(predictions-1)), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISP[r, g], ISR[r, g], OOSP[r, g], OOSR[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	end
end

runCounter = collect(0:testRuns)
varArr = vcat("variance",varianceTrainArray)
ISR
gammaArray'
combinedArray = vcat(round.(gammaArray',3), ISR)
ISRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISRsquaredMonthlyOther105.CSV", ISRcomb,",")

combinedArray = vcat(round.(gammaArray',3), ISP)
ISPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("ISPMonthlyOther105.CSV", ISPcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSR)
OOSRcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSRsquaredMonthlyOther105.CSV", OOSRcomb,",")

combinedArray = vcat(round.(gammaArray',3), OOSP)
OOSPcomb = hcat(runCounter, varArr, combinedArray)
writedlm("OOSPMonthlyOther105.CSV", OOSPcomb,",")
