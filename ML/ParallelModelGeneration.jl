using JuMP
using Gurobi
env = Gurobi.Env()

function generateLassoModel(Xtrain, Ytrain, gamma)
	bCols = size(Xtrain)[2]
	M = JuMP.Model(solver = GurobiSolver(env, OutputFlag = 0, Threads=(nprocs()-1)))
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

#	println("solved by worker $(myid())")
	return getvalue(b)

#	return M
end

function lassoValidationAndTest(gamma, xTrain, yTrain, xVali, yVali, xTest, yTest)
	beta = generateLassoModel(xTrain, yTrain, gamma)

	for i in 1:length(beta)
		if beta[i] <= 1e-6
			if beta[i] >= -1e-6
				beta[i] = 0
			end
		end
	end

	k = countnz(beta)
	if k == 0
		maxPairwise = 0
	else
		maxPairwise = findMaxCor(xTrain, beta)
	end

	valiRsqr = getRSquared(xVali, yVali, beta)
	testRsqr = getRSquared(xTest, yTest, beta)

	return maxPairwise, testRsqr, valiRsqr, k, beta
end

function processOutput(Xtrain, Ytrain, Xpred, Ypred, bSolved)
	for i in 1:length(bSolved)
		if bSolved[i] <= 1e-6
			if bSolved[i] >= -1e-6
				bSolved[i] = 0
			end
		end
	end

	#In-Sample R-squared value
	errors = (Ytrain-Xtrain*bSolved)
	errorTotal = sum(errors[i]^2 for i=1:length(errors))
	errorsMean = Ytrain-mean(Ytrain)
	errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
	ISRsquared = 1-(errorTotal/errorMeanTotal)

	#Indicator Results
	YpredValue = Ypred[1]
	Yestimate = Xpred'*bSolved
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

function generatSolveAndProcess(Xtrain, Ytrain, Xpred, Ypred, gamma)
	bSolved = generateLassoModel(Xtrain, Ytrain, gamma)

	ISRsquared, Indicator, YestimateValue = processOutput(Xtrain, Ytrain, Xpred, Ypred, bSolved)

	return ISRsquared, Indicator, YestimateValue#, bSolved
end
