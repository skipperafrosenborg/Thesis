#LASSO
using StatsBase
using DataFrames
using CSV

trainingSizeInput = parse(Int64, ARGS[1])
println(typeof(trainingSizeInput))

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
cd(path)
@everywhere include("ParallelModelGeneration.jl")
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")
#path = "$(homedir())/Documents/GitHub/Thesis/Data"

function runLassos(VIX, raw, expTrans, timeTrans, TA, trainingSize)
	#Skipper's path
	#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/"
	path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data/"
	#mainData = loadIndexDataNoDur(path)
	#mainData = loadIndexDataNoDurVIX(path)
	mainData = loadIndexDataNoDurLOGReturn(path)

	#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/"
	path = "/zhome/9f/d/88706/SpecialeCode/"
	dateAndReseccion = Array(mainData[:,end-1:end])
	mainDataArr = Array(mainData[:,1:end-2])
	mainDataArr[1,end]

	colNames = names(mainData)

	nRows = size(mainDataArr)[1]
	nCols = size(mainDataArr)[2]

	#fileName = path*"/Results/IndexData/LassoTests/240-1/2401_Shrink_"
	#fileName = path*"/Results/Test/Parallel/"
	fileName = path*"Results/IndexData/LassoTest/"*string(trainingSize)*"-1/"*string(trainingSize)*"1_Shrink_"

	#Reset HPC path
	#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
	#cd(path)

	mainXarr = mainDataArr[:,1:nCols-1]
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
	predictions = 1
	testRuns = nRows-trainingSize-predictions

	ISR = SharedArray{Float64}(testRuns, nGammas)
	Indi = SharedArray{Float64}(testRuns, nGammas)
	gammaArray = logspace(0, 3, nGammas)
	predictedArr = SharedArray{Float64}(testRuns, nGammas)
	realArr = SharedArray{Float64}(testRuns, 1)
	bSolvedArr = SharedArray{Float64}(testRuns, bCols)


	xTrainInput = [allData[r:(trainingSize+r), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
	YtrainInput = [allData[r:(trainingSize+r), bCols+1] for r = 1:(nRows-trainingSize-predictions)]
	XpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), 1:bCols] for r = 1:(nRows-trainingSize-predictions)]
	YpredInput  = [allData[(trainingSize+r+1):(trainingSize+r+predictions), bCols+1] for r = 1:(nRows-trainingSize-predictions)]

	for i in 1:nGammas
		gammaInput = [gammaArray[i] for r = 1:(nRows-trainingSize-predictions)]

		#Loop over gamma
		results = pmap(generatSolveAndProcess, xTrainInput, YtrainInput, XpredInput, YpredInput, gammaInput)

		for j in 1:testRuns
			ISR[j,i] = results[j][1]
			Indi[j,i] = results[j][2]
			predictedArr[j,i] = results[j][3]
			realArr[j,1] = YpredInput[j][1]
		end
	end

	combinedArray = vcat((round.(gammaArray,3))', ISR)
	dateAndReseccionOutput = dateAndReseccion[trainingSize+1+1:trainingSize+testRuns+predictions,:]
	dateAndReseccionOutput = vcat([1,2]',dateAndReseccionOutput)

	runCounter = collect(0:testRuns)
	ISRcomb = hcat(runCounter,dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"ISRsquared.CSV", ISRcomb,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"Indi.CSV", Indicomb,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"real.CSV", realArr,",")

	combinedArray = vcat((round.(gammaArray,3))', Indi)
	Indicomb = hcat(runCounter, dateAndReseccionOutput, combinedArray)
	writedlm(fileName*"predicted.CSV", predictedArr,",")

	#writedlm(fileName*"bSolved.CSV", bSolvedArr,",")
end

#@time(
	#Testing raw dataset
for timeTrans = 0:1
	for expTrans = 0:1
		for TA = 0:1
			runLassos(0, 1, timeTrans, expTrans, TA, trainingSizeInput)
		end
	end
end

#Testing macro dataset, including and excluding VIX
for VIX = 0:1
	for timeTrans = 0:1
		for expTrans = 0:1
			for TA = 0:1
				runLassos(VIX, 0, timeTrans, expTrans, TA, trainingSizeInput)
			end
		end
	end
end
#)

#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/ML"
#cd(path)
println("Finished")
