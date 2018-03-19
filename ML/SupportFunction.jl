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
	println("Transforming with t-3, t-6 and t-12")
	#the first row is -1 already, so it will be row +2, +5 and +11
	expandedX = copy(Array{Float64}(X))
	xCols = size(expandedX)[2]
	xRows = size(expandedX)[1]

	# -3 time series
	for i=1:xCols
		insertArray = []
		for j=1:(xRows-2)
			push!(insertArray, expandedX[j+2,i])
		end
		for j=(xRows-1):(xRows)
			push!(insertArray, 0)
		end
		expandedX = hcat(expandedX, insertArray)
	end

	# -6 time series
	for i=1:xCols
		insertArray = []
		for j=1:(xRows-5)
			push!(insertArray, expandedX[j+5,i])
		end
		for j=(xRows-4):(xRows)
			push!(insertArray, 0)
		end
		expandedX = hcat(expandedX, insertArray)
	end

	# -12 time series
	for i=1:xCols
		insertArray = []
		for j=1:(xRows-11)
			push!(insertArray, expandedX[j+11,i])
		end
		for j=(xRows-10):(xRows)
			push!(insertArray, 0)
		end
		expandedX = hcat(expandedX, insertArray)
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
	for j=1:(xRows-2)
		push!(MAs2arr, sum(Y[s] for s=(j+1):(j+2))/2)
	end
	for j=(xRows-1):(xRows)
		push!(MAs2arr,0)
	end

	#s=3 would be the average of the last 3 observations
	MAs3arr = []
	for j=1:(xRows-3)
		push!(MAs3arr, sum(Y[s] for s=(j+1):(j+3))/3)
	end
	for j=(xRows-2):(xRows)
		push!(MAs3arr,0)
	end

	#l=9 would be the average of the last 9 observations
	MAl9arr = []
	for j=1:(xRows-9)
		push!(MAl9arr, sum(Y[s] for s=(j+1):(j+9))/9)
	end
	for j=(xRows-8):(xRows)
		push!(MAl9arr,0)
	end

	#l=12 would be the average of the last 12 observations
	MAl12arr = []
	for j=1:(xRows-12)
		push!(MAl12arr, sum(Y[s] for s=(j+1):(j+12))/12)
	end
	for j=(xRows-11):(xRows)
		push!(MAl12arr,0)
	end

	#Compare s=1 and l=9 and create a column of buy signals
	#then do the same for s=2, 3 vs l=9
	#then l=12 vs. s=1,2,3
	MAMatrix = zeros(xRows, 6)
	#check s=1 and l=9
	for j=1:xRows
		if Y[j] >= MAl9arr[j]
			MAMatrix[j, 1] = 1
		end
	end
	#check s=2 and l=9
	for j=1:xRows
		if MAs2arr[j] >= MAl9arr[j]
			MAMatrix[j, 2] = 1
		end
	end
	#check s=3 and l=9
	for j=1:xRows
		if MAs3arr[j] >= MAl9arr[j]
			MAMatrix[j, 3] = 1
		end
	end

	#check s=1 and l=9
	for j=1:xRows
		if Y[j] >= MAl12arr[j]
			MAMatrix[j, 4] = 1
		end
	end
	#check s=2 and l=9
	for j=1:xRows
		if MAs2arr[j] >= MAl12arr[j]
			MAMatrix[j, 5] = 1
		end
	end
	#check s=3 and l=9
	for j=1:xRows
		if MAs3arr[j] >= MAl12arr[j]
			MAMatrix[j, 6] = 1
		end
	end

	expandedX = hcat(expandedX, MAMatrix)


	# MOMENTUM
	# If a stock is higher then it was m periods ago, it has MOMENTUM
	# computing for m=9
	Mom9arr = []
	for j=1:(xRows-9)
		if Y[j] > Y[j+9]
			push!(Mom9arr, 1)
		else
			push!(Mom9arr, 0)
		end
	end
	for j=(xRows-8):(xRows)
		push!(Mom9arr,0)
	end

	expandedX = hcat(expandedX, Mom9arr)

	# computing for m=9
	Mom12arr = []
	for j=1:(xRows-12)
		if Y[j] > Y[j+12]
			push!(Mom12arr, 1)
		else
			push!(Mom12arr, 0)
		end
	end
	for j=(xRows-11):(xRows)
		push!(Mom12arr,0)
	end

	expandedX = hcat(expandedX, Mom12arr)

	#Ensure we return a Float64 array
	expandedX = copy(Array{Float64}(expandedX))
	return expandedX
end


"""
Function that returns all the column names including ^2, log(), sqrt()
"""
function expandedColNamesToString(colNames)
	masterString = ""
	for i=1:length(colNames)-1
		masterString = masterString*string(colNames[i])*","
	end

	for i=1:length(colNames)-1
		masterString = masterString*string(colNames[i])*"^2,"
	end

	for i=1:length(colNames)-1
		masterString = masterString*"ln("*string(colNames[i])*"),"
	end

	for i=1:length(colNames)-1
		masterString = masterString*"sqrt("*string(colNames[i])*"),"
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
	for i=1:sampleSize
		sampleRows = selectSampleRowsWR(rowsPerSample, nRows)
		sampleX = createSampleX(standX, sampleRows)
		sampleY = createSampleY(standY, sampleRows)
		bSample[i,:] = solveForBeta(sampleX, sampleY, k, gamma, allCuts, bestZ);
		#bSample[i,:] = solveForBetaClosedForm(sampleX, sampleY, k, bestZ)
		if i == floor(sampleSize/2)
			println("50% through samples, $i missing")
		end
	end
	println("100% through samples")
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

	#fixSolution TO BE IMPLEMENTED
	fixSolution(TempStage2Model, bestZ)

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

function fixSolution(model, bestZ)
	if !isempty(bestZ)
		curUB = Gurobi.getconstrUB(model)
		curLB = Gurobi.getconstrLB(model)
		cutCols = size(bestZ)[1]
		curUB[4*bCols+1+HCPairCounter+(nCols)+1] = countnz(bestZ)
		curLB[4*bCols+1+HCPairCounter+(nCols)+1] = countnz(bestZ)
		Gurobi.setconstrUB!(model, curUB)
		Gurobi.setconstrLB!(model, curLB)
		for c = 1:(cutCols)
			Gurobi.changecoeffs!(model, [4*bCols+1+HCPairCounter+(nCols)+1], [bCols+c], [bestZ[c]])
		end
	end
	Gurobi.updatemodel!(model)
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
