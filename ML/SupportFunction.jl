"""
Function that returns the zscore column zscore for each column in the Matrix.
Must have at least two columns
"""
function zScoreByColumn(X)
	standX = Array{Float64}(X)
	for i=1:size(X)[2]
		temp = zscore(X[:,i])
		if any(isnan.(temp))
			println("Column $i remains unchanged as std = 0")
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
function shrinkValuesH(betaVector, kMax)
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
function gradDecent(X, y, L, epsilon, kMax)
	betaVector = shrinkValuesH(rand(size(X)[2]),kmax)

	#Initialise parameters
	iter = 0
	oldBetaVector = copy(betaVector)
	curError = twoNormRegressionError(X, y, betaVector) - twoNormRegressionError(X, y, oldBetaVector) + 10000

	while (iter < 10000 && curError > epsilon)
		oldBetaVector = copy(betaVector)

		#Calculate delta(g(beta))
		gradBeta = -X'*(y-X*betaVector)

		#Shrink smalles values
		betaVector = copy(shrinkValuesH(betaVector-1/L*gradBeta, kMax))

		curError = abs.(twoNormRegressionError(X, y, oldBetaVector) - twoNormRegressionError(X, y, betaVector))
		#println(curError)
		iter += 1
		if iter%1000 == 0
			#println("Iteration $iter with error on $curError")
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
    SSres = sum((y[i] - sum(X[i,j]*betaSolve[j] for j=1:size(X)[2]))^2 for i=1:length(y))
    SSTO = sum((y[i]-mean(y))^2 for i=1:length(y))
    Rsquared = 1-(SSres)/SSTO
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
	for i=1:xCols
		# squared transformation column
		insertArray = []
		for i=1:xRows
			push!(insertArray, expandedX[i,1]^2)
		end
		expandedX = hcat(expandedX, insertArray)
		# log transformation column
		insertArray = []
		for i=1:xRows
			push!(insertArray, log(expandedX[i,1]))
		end
		expandedX = hcat(expandedX, insertArray)
		# sqrt transformation column
		insertArray = []
		for i=1:xRows
			push!(insertArray, sqrt(expandedX[i,1]))
		end
		expandedX = hcat(expandedX, insertArray)
	end
	expandedX = copy(Array{Float64}(expandedX))
	return expandedX
end
