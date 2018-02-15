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
#NEED TO IMPLEMENT THAT QUADRATIC, LOG AND SQRT CAN'T BE SELECTED
function shrinkValuesH(betaVector, kMax, HCArray)
	#Make aboslute copy of betaVector
	absCopy = copy(abs.(betaVector))
	#Create zeroVector
	zeroVector = zeros(betaVector)

    #If any found, set value to 0 and continue

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
function gradDecent(X, y, L, epsilon, kMax, HC)
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

    betaVector = shrinkValuesH(rand(size(X)[2]),kmax, HCArray)

	#Initialise parameters
	iter = 0
	oldBetaVector = copy(betaVector)
	curError = twoNormRegressionError(X, y, betaVector) - twoNormRegressionError(X, y, oldBetaVector) + 10000

	while (iter < 10000 && curError > epsilon)
		oldBetaVector = copy(betaVector)

		#Calculate delta(g(beta))
		gradBeta = -X'*(y-X*betaVector)

		#Shrink smalles values
		betaVector = copy(shrinkValuesH(betaVector-1/L*gradBeta, kMax, HCArray))

		curError = abs.(twoNormRegressionError(X, y, oldBetaVector) - twoNormRegressionError(X, y, betaVector))
		iter += 1

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
			if count(k->(k<=0), expandedX[j,i]) > 0
				push!(insertArray, 0)
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
