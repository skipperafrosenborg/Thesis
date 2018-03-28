#LASSO
using JuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
using Bootstrap #External packages, must be added
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
cd(path)
@everywhere include("ParallelModelGeneration.jl")
@everywhere include("SupportFunction.jl")
@everywhere include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

# loopArr row = iteration
# boolean column: raw, time, exp, ta
#loopArr = zeros(1,4)

raw = 1
timeTrans = 1
TA = 1
expTrans = 1

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
#mainData = loadIndexDataNoDur(path)
mainData = loadIndexDataNoDurVIX(path)
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

#fileName = path*"/Results/IndexData/LassoTests/240-1/2401_Shrink_"
fileName = path*"/Test/Parallel/"
#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

mainXarr = mainDataArr[:,1:nCols-1]
if raw == 1
	fileName = fileName*"Raw"
	mainXarr = mainXarr[:,1:10]
else
	fileName = fileName*"Macro"
end

mainYarr = mainDataArr[:, nCols:nCols]

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
standY =mainDataArr[:, nCols:nCols]
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \nSolving Convex \n \n \n")

fileName = fileName*"_"

println(fileName)

nGammas = 10
trainingSize = 12
predictions = 1
testRuns = nRows-trainingSize-predictions
testRuns = 12
ISR = SharedArray{Float64}(testRuns, nGammas)
OOSR = SharedArray{Float64}(testRuns, nGammas)
Indi = SharedArray{Float64}(testRuns, nGammas)
gammaArray = logspace(0, 3, nGammas)
predictedArr = SharedArray{Float64}(testRuns, nGammas)
realArr = SharedArray{Float64}(testRuns, 1)
bSolvedArr = SharedArray{Float64}(testRuns, bCols)
a = SharedArray{Float64}(12)

xTrainInput = [allData[r:(trainingSize+r), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
YtrainInput = [allData[r:(trainingSize+r), bCols+1] for r = 1:(nRows-trainingSize-predictions)]
XpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
YpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1] for r = 1:(nRows-trainingSize-predictions)]
gammaInput = [1 for r = 1:(nRows-trainingSize-predictions)]

test = pmap(generatSolveAndProcess, xTrainInput, YtrainInput, XpredInput, YpredInput, gammaInput)

function runLassos(raw, timeTrans, expTrans, TA)
	#Skipper's path
	path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
	#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
	#mainData = loadIndexDataNoDur(path)
	mainData = loadIndexDataNoDurVIX(path)
	mainDataArr = Array(mainData)

	colNames = names(mainData)

	nRows = size(mainDataArr)[1]
	nCols = size(mainDataArr)[2]

	#fileName = path*"/Results/IndexData/LassoTests/240-1/2401_Shrink_"
	fileName = path*"/Test/Parallel/"
	#Reset HPC path
	#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
	#cd(path)

	mainXarr = mainDataArr[:,1:nCols-1]
	if raw == 1
		fileName = fileName*"Raw"
		mainXarr = mainXarr[:,1:10]
	else
		fileName = fileName*"Macro"
	end

	mainYarr = mainDataArr[:, nCols:nCols]

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
	standY =mainDataArr[:, nCols:nCols]
	allData = hcat(standX, standY)
	bCols = size(standX)[2]
	nRows = size(standX)[1]
	println(" \n \n \nSolving Convex \n \n \n")

	fileName = fileName*"_"

	println(fileName)

	#=
	@everywhere function solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
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

		for i in 1:length(bSolved)
			if bSolved[i] <= 1e-6
				if bSolved[i] >= -1e-6
					bSolved[i] = 0
				end
			end
		end

		k = countnz(bSolved)

		#In-Sample R-squared value
		errors = (Ytrain-Xtrain*bSolved)
		errorTotal = sum(errors[i]^2 for i=1:length(errors))
		errorsMean = Ytrain-mean(Ytrain)
		errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
		ISRsquared = 1-(errorTotal/errorMeanTotal)

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


		return ISRsquared, Indicator, YestimateValue#, bSolved
	end
	=#

	nGammas = 10
	trainingSize = 12
	predictions = 1
	testRuns = nRows-trainingSize-predictions
	testRuns = 12
	ISR = SharedArray{Float64}(testRuns, nGammas)
	OOSR = SharedArray{Float64}(testRuns, nGammas)
	Indi = SharedArray{Float64}(testRuns, nGammas)
	gammaArray = logspace(0, 3, nGammas)
	predictedArr = SharedArray{Float64}(testRuns, nGammas)
	realArr = SharedArray{Float64}(testRuns, 1)
	bSolvedArr = SharedArray{Float64}(testRuns, bCols)
	a = SharedArray{Float64}(12)

	xTrainInput = [allData[r:(trainingSize+r), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
	YtrainInput = [allData[r:(trainingSize+r), bCols+1] for r = 1:(nRows-trainingSize-predictions)]
	XpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
	YpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1] for r = 1:(nRows-trainingSize-predictions)]



	result = @sync @parallel for r = 1:(nRows-trainingSize-predictions)
		println("Processor ", myid(), " is processing job ", r)
		Xtrain = allData[r:(trainingSize+r), 1:bCols]
		Ytrain = allData[r:(trainingSize+r), bCols+1]
		Xpred  = allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols]
		Ypred  = allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1]
		realArr[r] = Ypred[1]

		#g=5
		for g = 1:nGammas
			gamma = gammaArray[g]
			#ISR[r, g], Indi[r, g], predictedArr[r, g], bSolvedArr[r,:] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
			ISR[r, g], Indi[r, g], predictedArr[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
		end

		if (r%100 == 0)
			println("Row $r/",(nRows-trainingSize-predictions))
		end
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

	combinedArray = vcat((round.(gammaArray,3))', ISR)


	runCounter = collect(0:testRuns)
	ISRcomb = hcat(runCounter, combinedArray)
	writedlm(fileName*"ISRsquared.CSV", ISRcomb,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, combinedArray)
	writedlm(fileName*"Indi.CSV", Indicomb,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, combinedArray)
	writedlm(fileName*"real.CSV", realArr,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, combinedArray)
	writedlm(fileName*"predicted.CSV", predictedArr,",")

	writedlm(fileName*"bSolved.CSV", bSolvedArr,",")
end

runLassos(1,1,1,1)

for raw = 0:1
	for timeTrans = 0:1
		for expTrans = 0:1
			for TA = 0:1
				runLassos(raw, timeTrans, expTrans, TA)
			end
		end
	end
end
println("Finished")

#=
#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
#mainData = loadIndexDataNoDur(path)
mainData = loadIndexDataNoDurVIX(path)
mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]
#mainXarr = mainXarr[:,1:10]

mainYarr = mainDataArr[:, nCols:nCols]

fileName = path*"/Results/IndexData/LassoTests/12-1 VIX/121_Shrink_MacroTime_"
println(fileName)
#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

# Transform with time elements
mainXarr = expandWithTime3612(mainXarr)

# Transform #
#mainXarr = expandWithTransformations(mainXarr)

#mainXarr = expandWithMAandMomentum(mainXarr, mainYarr, (nCols-1))

# Standardize
standX = zScoreByColumn(mainXarr)
standY =mainDataArr[:, nCols:nCols]
allData = hcat(standX, standY)
bCols = size(standX)[2]
nRows = size(standX)[1]
println(" \n \n \nSolving Convex \n \n \n")

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

	for i in 1:length(bSolved)
		if bSolved[i] <= 1e-6
			if bSolved[i] >= -1e-6
				bSolved[i] = 0
			end
		end
	end

	k = countnz(bSolved)

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

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


	return ISRsquared, Indicator, YestimateValue#, bSolved
end

nGammas = 10
trainingSize = 12
predictions = 1
testRuns = nRows-trainingSize-predictions
ISR = zeros(testRuns, nGammas)
OOSR = zeros(testRuns, nGammas)
Indi = zeros(testRuns, nGammas)
gammaArray = logspace(0, 3, nGammas)
predictedArr = zeros(testRuns, nGammas)
realArr = zeros(testRuns, 1)
#bSolvedArr = zeros(testRuns, bCols)

for r = 1:(nRows-trainingSize-predictions)
	Xtrain = allData[r:(trainingSize+r), 1:bCols]
	Ytrain = allData[r:(trainingSize+r), bCols+1]
	Xpred  = allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols]
	Ypred  = allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1]
	realArr[r] = Ypred[1]

	#ISR[r, g], OOSR[r, g], Indi[r, g], predictedArr[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	#ISR[r, g], Indi[r, g], predictedArr[r, g], bSolvedArr[r,:] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	#g=2
	for g = 1:nGammas
		gamma = gammaArray[g]
		#ISR[r, g], OOSR[r, g], Indi[r, g], predictedArr[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
		ISR[r, g], Indi[r, g], predictedArr[r, g] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
	end

	if (r%100 == 0)
		println("Row $r/",(nRows-trainingSize-predictions))
	end
end

combinedArray = vcat((round.(gammaArray,3))', ISR)

runCounter = collect(0:testRuns)
ISRcomb = hcat(runCounter, combinedArray)
writedlm(fileName*"ISRsquared.CSV", ISRcomb,",")

combinedArray = vcat((round.(gammaArray,3))', Indi)
Indicomb = hcat(runCounter, combinedArray)
writedlm(fileName*"Indi.CSV", Indicomb,",")

combinedArray = vcat((round.(gammaArray,3))', Indi)
Indicomb = hcat(runCounter, combinedArray)
writedlm(fileName*"real.CSV", realArr,",")

combinedArray = vcat((round.(gammaArray,3))', Indi)
Indicomb = hcat(runCounter, combinedArray)
writedlm(fileName*"predicted.CSV", predictedArr,",")

#writedlm(fileName*"bSolved.CSV", bSolvedArr,",")


println("Finished")
=#
