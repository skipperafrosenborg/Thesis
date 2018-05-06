"""
findMaxCor(standX, bestSolution)
Returns the maximum correlation form the selected variables
"""
function findMaxCor(standX, bestSolution)
	chosenVar = find(bestSolution)
	for i = chosenVar
		if bestSolution[i] < 1e-6 && bestSolution[i] > -1e-6
			bestSolution[i] = 0
		end
	end
	chosenVar = find(bestSolution)
	return maximum(cor(standX[:,chosenVar])-Diagonal(cor(standX[:,chosenVar])))
end

"""
Function that returns the zscore column zscore for each column in the Matrix.
Must have at least two columns
"""
function zScoreByColumn(X)
	standX = Array{Float64}(X)
	for i=1:size(X)[2]
		temp = zscore(X[:,i])
		if any(isnan.(temp))
			#println("Column $i remains unchanged as std = 0")
		else
			standX[:,i] = copy(zscore(X[:,i]))
		end
	end
	return standX
end

"""
Function an array with the kMax largest absolute values
and all other values shrunk to 0
"""
function shrinkValuesH(betaVector, kMax, HCArray)
	#Make aboslute copy of betaVector
	absCopy = copy(abs.(betaVector))
	#Create zeroVector
	zeroVector = zeros(betaVector)

	for i in 1:kMax
		#Find index of maximum value in absolute vector
		ind = indmax(absCopy)

		#Replace index in 0 vector with betaVector value of index
		zeroVector[ind] = betaVector[ind]

		#Replace index in absolute vector with 0
		absCopy[ind] = 0

        #Check if index is in HC array and add all
        HCvar = find(HCArray[:,1] .== ind)

        if !isempty(HCvar)
            for j in 1:length(HCvar)
                absCopy[HCArray[HCvar[j],2]] = 0
                #println("Extra shrink")
            end
        end

		#Find place in column
		nColOrig = size(zeroVector)[1]/4

		if ind%nColOrig == 0
			indNum = nColOrig
		else
			indNum = ind%nColOrig
		end
		placeInColumn = floor(ind/nColOrig)
		#println(nColOrig)
		#println(placeInColumn)
		#println(ind%nColOrig)
		for j in 0:3
			if j != placeInColumn
				absCopy[Int64(indNum+j*nColOrig)] = 0
			end
		end
	end
	return zeroVector
end

"""
Function to calculate two norm regression error
"""
function twoNormRegressionError(X, y, betaVector)
	return 1/2*norm(y-X*betaVector)^2
end

"""
Vanilla gradient decent which only keeps the kMax biggest values. All other
values are shrunk to 0.
"""
function gradDecent(X, y, L, epsilon, kMax, HC, bSolved)
    HCArray = Matrix(0,2)
    rho = 0.8
    for k=1:size(X)[2]
    	for j=1:size(X)[2]
    		if k != j
    			if HC[k,j] >= rho
                    HCArray = cat(1, HCArray, [k j])
    			end
    		end
    	end
    end

	if countnz(bSolved) < 1
		bSolved = rand(size(X)[2])
	end

    betaVector = shrinkValuesH(bSolved,kmax, HCArray)

	#Initialise parameters
	iter = 0
	oldBetaVector = copy(betaVector)
	curError = twoNormRegressionError(X, y, betaVector) - twoNormRegressionError(X, y, oldBetaVector) + 10000

	while (iter < 10000 && curError > epsilon)
		oldBetaVector = copy(betaVector)

		#Calculate delta(g(beta))
		gradBeta = X'*(y-X*betaVector)

		#Shrink smalles values
		betaVector = copy(shrinkValuesH(betaVector+1/L*gradBeta, kMax, HCArray))

		curError = abs.(twoNormRegressionError(X, y, oldBetaVector) - twoNormRegressionError(X, y, betaVector))
		iter += 1

		#println("Iteration $iter, error $curError")

		if iter%1000 == 0
			println("Iteration $iter with error on $curError")
		end
	end
	#println("End error: $curError")
	#println("Number of iterations: $iter")
	return  betaVector
end

"""
Function to that returns the R^2 value
"""
function getRSquared(X,y,betaSolve)
    SSres = sum((y[i] - X[i,:]'*betaSolve)^2 for i=1:length(y))
    SSTO = sum((y[i]-mean(y))^2 for i=1:length(y))
    Rsquared = 1-SSres/SSTO
    return Rsquared
end

function getRMSE(X,y,betaSolve)
    SSres = sum((y[i] - X[i,:]'*betaSolve)^2 for i=1:length(y))
	n=length(y)
	RMSE = sqrt(SSres/n)
	return RMSE
end


"""
Function that prints the non zero solutions values of beta
"""
function printNonZeroValues(bSolved)
    for i=1:length(bSolved)
        if !isequal(bSolved[i], 0)
            println("b[$i] = ", bSolved[i])
        end
    end
end

"""
Transforming X as: Squared, log & square-root
"""
function expandWithTransformations(X)
	println("Transforming as: Squared, log & square-root")
	expandedX = copy(Array{Float64}(X))
	xCols = size(expandedX)[2]
	xRows = size(expandedX)[1]

	# squared transformation column
	for i=1:xCols
		insertArray = []
		for j=1:xRows
			push!(insertArray, expandedX[j,i]^2)
		end
		expandedX = hcat(expandedX, insertArray)
	end

	# Natural log transformation column
	for i=1:xCols
		insertArray = []
		for j=1:xRows
			if count(k->(k<0), expandedX[j,i]) > 0
				push!(insertArray, 0)
			elseif count(k->(k==0), expandedX[j,i]) > 0
				push!(insertArray, log(expandedX[j,i]+0.00001))
			else
				push!(insertArray, log(expandedX[j,i]))
			end
		end
		expandedX = hcat(expandedX, insertArray)
	end

	# sqrt transformation column
	for i=1:xCols
		insertArray = []
		for j=1:xRows
			if count(k->(k<=0), expandedX[j,i]) > 0
				push!(insertArray, 0)
			else
				push!(insertArray, sqrt(expandedX[j,i]))
			end
		end
		expandedX = hcat(expandedX, insertArray)
	end

	#Ensure we return a Float64 array
	expandedX = copy(Array{Float64}(expandedX))
	return expandedX
end

"""
Transforming X with time elements -3, -6 and -12
"""
function expandWithTime3612(X)
	println("Transforming with t-1, t-3, ..., t-12")
	expandedX = copy(Array{Float64}(X))
	xCols = size(expandedX)[2]
	xRows = size(expandedX)[1]


	for k=1:12
		# k = how many month we look back: -1:-12 time series
		for i=1:xCols
			# i loops over all columns in the X matrix
			insertArray = []

			# j loops over rows in X matrix
			for j=1:k
				# adding 0-rows for the first k rows as we do not have any
				# previous observations
				push!(insertArray, 0)
			end

			for j=k+1:xRows
				push!(insertArray, expandedX[j-k,i])
			end
			expandedX = hcat(expandedX, insertArray)
		end
	end

	#Ensure we return a Float64 array
	expandedX = copy(Array{Float64}(expandedX))
	return expandedX
end



"""
Transforming X with moving average and momentum signals
"""
function expandWithMAandMomentum(X, Y, originalColumns)
	println("Transforming with momentum (9, 12) and MA(1,2,3 & 9,12) ")
	#the first row is -1 already, so it will be row +2, +5 and +11
	expandedX = copy(Array{Float64}(X))
	xCols = originalColumns
	xRows = size(expandedX)[1]
	#Already have moving average for s=1, since that our base row.

	#s=2 would be the average of the last 2 observations
	MAs2arr = []
	for j=1:2
		push!(MAs2arr,0)
	end
	for j=1:(xRows-2)
		push!(MAs2arr, sum(Y[s] for s=(j):(j+1))/2)
	end

	#s=3 would be the average of the last 3 observations
	MAs3arr = []
	for j=1:3
		push!(MAs3arr,0)
	end
	for j=1:(xRows-3)
		push!(MAs3arr, sum(Y[s] for s=(j):(j+2))/3)
	end

	#l=9 would be the average of the last 9 observations
	MAl9arr = []
	for j=1:9
		push!(MAl9arr,0)
	end
	for j=1:(xRows-9)
		push!(MAl9arr, sum(Y[s] for s=(j):(j+8))/9)
	end

	#l=12 would be the average of the last 12 observations
	MAl12arr = []
	for j=1:12
		push!(MAl12arr,0)
	end
	for j=1:(xRows-12)
		push!(MAl12arr, sum(Y[s] for s=(j):(j+11))/12)
	end

	#Compare s=1 and l=9 and create a column of buy signals
	#then do the same for s=2, 3 vs l=9
	#then l=12 vs. s=1,2,3
	MAMatrix = zeros(xRows, 6)
	#check s=1 and l=9
	for j=10:xRows
		if Y[j-1] >= MAl9arr[j]
			MAMatrix[j, 1] = 1
		end
	end
	#check s=2 and l=9
	for j=10:xRows
		if MAs2arr[j] >= MAl9arr[j]
			MAMatrix[j, 2] = 1
		end
	end
	#check s=3 and l=9
	for j=10:xRows
		if MAs3arr[j] >= MAl9arr[j]
			MAMatrix[j, 3] = 1
		end
	end

	#check s=1 and l=12
	for j=13:xRows
		if Y[j-1] >= MAl12arr[j]
			MAMatrix[j, 4] = 1
		end
	end

	#check s=2 and l=12
	for j=13:xRows
		if MAs2arr[j] >= MAl12arr[j]
			MAMatrix[j, 5] = 1
		end
	end
	#check s=3 and l=12
	for j=13:xRows
		if MAs3arr[j] >= MAl12arr[j]
			MAMatrix[j, 6] = 1
		end
	end

	expandedX = hcat(expandedX, MAMatrix)


	# MOMENTUM
	# If a stock is higher then it was m periods ago, it has MOMENTUM
	# computing for m=9
	Mom9arr = []
	for j=1:10
		push!(Mom9arr,0)
	end
	for j=11:xRows
		if Y[j-1] > Y[j-10]
			push!(Mom9arr, 1)
		else
			push!(Mom9arr, 0)
		end
	end

	expandedX = hcat(expandedX, Mom9arr)

	# computing for m=12
	Mom12arr = []
	for j=1:13
		push!(Mom12arr,0)
	end
	for j=14:xRows
		if Y[j-1] > Y[j-13]
			push!(Mom12arr, 1)
		else
			push!(Mom12arr, 0)
		end
	end

	expandedX = hcat(expandedX, Mom12arr)

	#Ensure we return a Float64 array
	expandedX = copy(Array{Float64}(expandedX))
	return expandedX
end

function expandColNamesTimeToString(featureNames)
	loopLength = length(featureNames)

	for j = 1:12
		for i=1:loopLength
			#println(Symbol(string(featureNames[i])*" t-"*string(j)))
			featureNames = vcat(featureNames, Symbol(string(featureNames[i])*" t-"*string(j)))
		end
	end

	return featureNames
end

"""
Function that returns all the column names including ^2, log(), sqrt()
"""
function expandedColNamesToString(colNames, ExpTrans, TA)
	masterString = ""

	if ExpTrans == 1
		for i=1:length(colNames)
			masterString = masterString*string(colNames[i])*","
		end

		for i=1:length(colNames)
			masterString = masterString*string(colNames[i])*"^2,"
		end

		for i=1:length(colNames)
			masterString = masterString*"ln("*string(colNames[i])*"),"
		end

		for i=1:length(colNames)
			masterString = masterString*"sqrt("*string(colNames[i])*"),"
		end
	end

	if TA == 1
		masterString = masterString * "MA 1-9,"
		masterString = masterString * "MA 2-9,"
		masterString = masterString * "MA 3-9,"
		masterString = masterString * "MA 1-12,"
		masterString = masterString * "MA 2-12,"
		masterString = masterString * "MA 3-12,"
		masterString = masterString * "Mo 9,"
		masterString = masterString * "MA 12"
	end

	return masterString
end

"""
Function to identify which parameters have been selected (transformed or not)
"""
function identifyParameters(betaSolution, colNames)
	originalColumns = Int64(length(betaSolution)/4)
	# test original parameters
	count = 1
	for i=1:originalColumns
		if !isequal(betaSolution[i],0)
			println("Parameter '", colNames[count],"' has been selected with value ", round(betaSolution[i],3))
		end
		count += 1
	end

	# test for squared
	count = 1
	for i=(originalColumns+1):(2*originalColumns)
		if !isequal(betaSolution[i],0)
			println("Parameter '", colNames[count],"^2' has been selected with value ", round(betaSolution[i],3))
		end
		count += 1
	end

	# test for natural log
	count = 1
	for i=(originalColumns*2+1):(3*originalColumns)
		if !isequal(betaSolution[i],0)
			println("Parameter 'ln(", colNames[count],")' has been selected with value ", round(betaSolution[i],3))
		end
		count += 1
	end
	# test for sqrt
	count = 1
	for i=(originalColumns*3+1):(4*originalColumns)
		if !isequal(betaSolution[i],0)
			println("Parameter 'sqrt(", colNames[count],")' has been selected with value ", round(betaSolution[i],3))
		end
		count += 1
	end
end

"""
Functions to make a split in data
"""
function createSampleX(x, inputRows)
	inputX = copy(x)
	outputX = inputX[inputRows,:]
	return outputX
end

function createSampleY(y, inputRows)
	inputY = copy(y)
	outputY = inputY[inputRows,:]
	return outputY
end

"""
Creates a sample of random rows without replacement.
Not useful for financial data, if we want a timeseries.
"""

function selectSampleRows(rowsWanted, nRows)
	rowsSelected = sample(1:nRows, rowsWanted, replace = false)
	return rowsSelected
end

function selectSampleRowsWR(rowsWanted, nRows)
	rowsSelected = rand(1:nRows, rowsWanted)
	return rowsSelected
end

function splitDataIn2(data, rowsWanted, nRows)
	rowsOne = selectSampleRows(rowsWanted, nRows)
	rowsTwo = []
	for i = 1:nRows
		if !(i in rowsOne)
			push!(rowsTwo, i)
		end
	end
	dataSplitOne = data[rowsOne, :]
	dataSplitTwo = data[rowsTwo, :]

	return dataSplitOne, dataSplitTwo
end

"""
Type that allows us to track solution progress (time, nodes searched, objective and bestbound)
"""
type NodeData
    time::Float64  # in seconds since the epoch
    node::Int
    obj::Float64
    bestbound::Float64
end

"""
Function that allows us to push information into the type NodeData
"""
function infocallback(cb)
	node      = MathProgBase.cbgetexplorednodes(cb)
	#println("INFO 1, nodes visited: ", node)
	obj       = MathProgBase.cbgetobj(cb)
	#println("INFO 2, objective is: ", obj)
	bestbound = MathProgBase.cbgetbestbound(cb)
	#println("INFO 3, best bound is: ", bestbound)
	push!(bbdata, NodeData(time_ns(),node,obj,bestbound))
end

"""
Function that converts a NodeData type into a csv file in the working directory
"""
function printSolutionToCSV(stringName, bbdata)
	open(stringName,"w") do fp
		println(fp, "time,node,obj,bestbound")
		for bb in bbdata
			println(fp, round(bb.time,4), ",", round(bb.node,2), ",",
						round(bb.obj,2), ",", round(bb.bestbound,2))
		end
	end
end

"""
Generating samplesize amount of beta-values for a beta distribution
"""
function createBetaDistribution(bSample, standX, standY, k, sampleSize, rowsPerSample, gamma, allCuts, bestZ)
	println("0% through samples")
	empty1 = 0
	empty2 = 0
	continueBool = false
	for i=1:sampleSize
		continueBool = false
		sampleRows = selectSampleRowsWR(rowsPerSample, nRows)
		sampleX = createSampleX(standX, sampleRows)
		sampleY = createSampleY(standY, sampleRows)
		HC = cor(sampleX) # this is added to support function below
		#println("Maximum pairwise correlation is: ",findMaxCor(standX, bestZ))

		bSample[i,:], empty1, empty2 = buildAndSolveStage2(sampleX, sampleY, k, gamma, false, [], allCuts, bestZ, HC)
		if empty1 == 9999
			newStandX = sampleX
			newStandY = sampleY
			newK = k
			newGamma = gamma
			newAllCuts = allCuts
			newBestZ = bestZ
			newHC = HC
			#println("\n\n\n STOP EVERYTHING \n\n\n")
		end

		#bSample[i,:] = solveForBeta(sampleX, sampleY, k, gamma, allCuts, bestZ); #This is replaced by above function
		#bSample[i,:] = solveForBetaClosedForm(sampleX, sampleY, k, bestZ)
		if i == floor(sampleSize/2)
			println("50% through samples, $i missing")
		end
	end
	println("100% through samples")
	return bSample
end

function solveForBetaClosedForm(X, Y, k, bestZ)
	tempX = zeros(size(X)[1],k)
	j=0
	for i in 1:length(bestZ)
		if bestZ[i] != 0
			j += 1
			tempX[:,j] = X[:,i]
		end
		if j == k break end
	end

	nonZeroBeta = inv(tempX'*tempX)*tempX'*Y
	bSolved=zeros(bCols)
	j=0
	for i in 1:length(bestZ)
		if bestZ[i] != 0
			j += 1
			bSolved[i] = nonZeroBeta[j]
		end
		if j == k break end
	end
	return bSolved
end


function solveForBeta(X, Y, k, gamma, allCuts, bestZ)
	TempStage2Model, HCPairCounter = buildStage2(X, Y, k);

	#Set kMax rhs constraint
	curUB = Gurobi.getconstrUB(TempStage2Model) #Get current UBbounds
	curUB[bCols*4+2] = k #Change upperbound in current bound vector
	Gurobi.setconstrUB!(TempStage2Model, curUB) #Push bound vector to model
	Gurobi.updatemodel!(TempStage2Model)

	#Set new Big M
	newBigM = 3#tau*norm(warmStartBeta, Inf)
	changeBigM(TempStage2Model,newBigM)

	changeGamma(TempStage2Model, gamma)

	#fixSolution
	fixSolution(TempStage2Model, bestZ, HCPairCounter)

	if !isempty(allCuts)
		addCuts(TempStage2Model, allCuts, 0)
	end
	#println(Gurobi.getconstrLB(TempStage2Model));
	#println(Gurobi.getconstrUB(TempStage2Model));

	#Solve Stage 2 model
	status = Gurobi.optimize!(TempStage2Model);
	#println("Objective value: ", Gurobi.getobjval(TempStage2Model))

	sol = Gurobi.getsolution(TempStage2Model);
	#Get solution and calculate R^2
	bSolved = sol[1:bCols]
	return bSolved
end

"""
Function that fixs the z variables to the same as previous to see if the
distribution around those values are constant.
"""
function fixSolution(model, bestZ, HCPairCounter)
	if !isempty(bestZ)
		curUB = Gurobi.getconstrUB(model)
		curLB = Gurobi.getconstrLB(model)
		cutCols = size(bestZ)[1]
		# Access the first psudo cut contraint (not active as we rebuild the model)
		# and sets this constraint to be == number of variables
		curUB[4*bCols+1+HCPairCounter+(nCols)+1] = countnz(bestZ)
		curLB[4*bCols+1+HCPairCounter+(nCols)+1] = countnz(bestZ)
		Gurobi.setconstrUB!(model, curUB)
		Gurobi.setconstrLB!(model, curLB)
		## Sets all active z varibles to 1 and all non active to 0 to ensure we pick these variables.
		for c = 1:(cutCols)
			Gurobi.changecoeffs!(model, [4*bCols+1+HCPairCounter+(nCols)+1], [bCols+c], [bestZ[c]])
		end
	end
	Gurobi.updatemodel!(model)
end

function changeBigM(model, newBigM)
	startIndx = bCols
	for i in 1:bCols
		Gurobi.changecoeffs!(model,[i],[startIndx+i],[-newBigM])
		Gurobi.changecoeffs!(model,[i+bCols],[startIndx+i],[-newBigM])
		Gurobi.updatemodel!(model)
	end
end

function changeGamma(model, newGamma)
	startRow  = bCols*4+1
	startIndx = bCols*2
	for i in 1:bCols
		Gurobi.changecoeffs!(model, [startRow], [startIndx+i], [newGamma])
		Gurobi.updatemodel!(model)
	end
end

function changeGammaLasso(model, newGamma)
	startRow  = bCols*2+1
	startIndx = bCols
	for i in 1:bCols
		Gurobi.changecoeffs!(model, [startRow], [startIndx+i], [newGamma])
		Gurobi.updatemodel!(model)
	end
end

function addCuts(model, cutMatrix, preCutCounter)
	if !isempty(cutMatrix)
		curUB = Gurobi.getconstrUB(model)
		cutRows = size(cutMatrix)[1]
		cutCols = size(cutMatrix)[2]
		for r = 1:cutRows
			curUB[4*bCols+1+HCPairCounter+(nCols)+r+preCutCounter+1] = cutMatrix[r, cutCols]-1
			Gurobi.setconstrUB!(model, curUB)
			for c = 1:(cutCols-1)
				Gurobi.changecoeffs!(model, [4*bCols+1+HCPairCounter+(nCols)+r+preCutCounter+1], [bCols+c], [cutMatrix[r,c]])
			end
		end
	end
	Gurobi.updatemodel!(model)
end


"""
Creates a confidence interval based on confidence interval level and nBoot
bootstrapped samples from the created beta distribution
"""
function createConfidenceIntervalArray(sampleInput, nBoot, confLevel)
	bSample = convert(Array{Float64}, sampleInput)
	bCols = size(bSample)[2]
	confIntArray = Matrix(2, bCols)
	for i=1:bCols
		bs = bootstrap(bSample[:,i], mean, BasicSampling(nBoot))
		cil = confLevel
		CiEst = Bootstrap.ci(bs, BasicConfInt(cil))
		confIntArray[1,i] = CiEst[1][2] #lower
		confIntArray[2,i] = CiEst[1][3]
	end
	return confIntArray
end
"""
Based on the confidence intervals created, significance of found parameters can
be tested with this function
"""
function testSignificance(confIntArray99, confIntArray95, confIntArray90, bResult)
	bCols = size(bResult)[1]
	significance = zeros(bCols)
	for i=1:bCols
		if bResult[i] >= confIntArray99[1,i] && bResult[i] <=confIntArray99[2,i] && confIntArray99[1,i] > 0
			significance[i] = 0.99
		elseif bResult[i] >= confIntArray99[1,i] && bResult[i] <=confIntArray99[2,i] && confIntArray99[2,i] < 0
			significance[i] = 0.99
		elseif bResult[i] >= confIntArray95[1,i] && bResult[i] <=confIntArray95[2,i] && confIntArray95[1,i] > 0
			significance[i] = 0.95
		elseif bResult[i] >= confIntArray95[1,i] && bResult[i] <=confIntArray95[2,i] && confIntArray95[2,i] < 0
			significance[i] = 0.95
		elseif bResult[i] >= confIntArray90[1,i] && bResult[i] <=confIntArray90[2,i] && confIntArray90[1,i] > 0
			significance[i] = 0.90
		elseif bResult[i] >= confIntArray90[1,i] && bResult[i] <=confIntArray90[2,i] && confIntArray90[2,i] < 0
			significance[i] = 0.90
		else
			significance[i] = 0
		end
	end
	return significance
end

#using StatPlots
#using Distributions
using HypothesisTests
function residualTesting(best3Beta, standX, standY)
    nModels = size(best3Beta)[1]
    bCols = size(standX)[2]
    nRows = size(standX)[1]
    residuals = Matrix(nRows, nModels)

    for i = 1:nModels
        residuals[:,i] = standY - standX*best3Beta[i, 4:bCols+3]
        res = convert(Array{Float64,1}, residuals[:,i])
        #plot(
        #    qqnorm(res, qqline = :R)
        #)
        BinomTest(res)
        JBTest(res)
    end
    writedlm("residuals.CSV", residuals,",")
end


function BinomTest(residuals)
	boo = false
	posCount=0
	for i=1:length(residuals)
		if residuals[i] > 0
			posCount+=1
		end
	end

	binomTest = BinomialTest(posCount, length(residuals), 0.5)
	if pvalue(binomTest) <= 0.05
		println("Binomial test (equal probability of + and -) failed, p=",pvalue(binomTest))
	else
		println("Binomial test is passed")
		boo = true
	end
	return boo
end

function JBTest(residuals)
    boo = false
    JBObject = JarqueBeraTest(residuals)
    if pvalue(JBObject) <= 0.05
		println("JarqueBera test (residual dist. similar to normal) failed, p=",pvalue(JBObject))
	else
		println("JarqueBera test is passed")
		boo = true
	end
    return boo
end



"""
SUPPORT FUNCTIONS FOR CEO AND LASSO MEAN VARIANCE
"""

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

    return period1NReturn, periodReturn, wStar, forecastRow
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
