using JuMP
using Gurobi

function generateLassoModel(Xtrain, Ytrain, gamma)
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

	return M
end

function solveModel(M::JuMP.Model, Xtrain)
	solve(M)
	

	return getvalue(b), "solved by worker $(myid())"
end

function processOutput(Xtrain, Ytrain, Xpred, Ypred, bSolved)
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

function generatSolveAndProcess(Xtrain, Ytrain, Xpred, Ypred, gamma)
	M = generateLassoModel(Xtrain, Ytrain, gamma)
	bSolved = solveModel(M)

	ISRsquared, Indicator, YestimateValue = processOutput(Xtrain, Ytrain, Xpred, Ypred, gamma)

	return ISRsquared, Indicator, YestimateValue#, bSolved
end
