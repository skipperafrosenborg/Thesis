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

#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
#mainData = loadConcrete(path)
#fileName = path*"/Results/Concrete/Concrete"
#mainData = loadHousingData(path)
#fileName = path*"/Results/HousingData/HousingData"
#mainData = loadCPUData(path)
#fileName = path*"/Results/CPUData/CPUData"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

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

#=


#Split total data into  train / vali / test
halfRows = Int64(floor(size(allData)[1]/2))
train, test = splitDataIn2(allData, halfRows, size(allData)[1])
halfRows = Int64(floor(size(test)[1]/2))
test, vali = splitDataIn2(test, halfRows, size(test)[1])

#Split data by X and y
standX = train[:,1:nCols-1]
standY = train[:,nCols]
standXTest = test[:,1:nCols-1]
standYTest = test[:,nCols]
standXVali = vali[:,1:nCols-1]
standYVali = vali[:,nCols]
=#

#=
mainData, testData = loadElevatorData(path)
dataSize = size(mainData)
colNames = names(mainData)
trainingData = Array(mainData)
testData = Array(testData)
nRows = size(trainingData)[1]
nCols = size(trainingData)[2]
=#

#=
# Output for Lasso regression in R
writedlm(fileName*"XTrain.CSV",standX,",")
writedlm(fileName*"YTrain.CSV",standY,",")
writedlm(fileName*"XTest.CSV",standXTest,",")
writedlm(fileName*"YTest.CSV",standYTest,",")
writedlm(fileName*"XVali.CSV",standXVali,",")
writedlm(fileName*"YVali.CSV",standYVali,",")
=#

#Initialise values for check later
bCols = size(standX)[2]
kValue = []
RsquaredValue = []
bSolved = []

function buildLasso(standX, standY)
	nRows = size(standX)[1]

	#Define parameters and model
	lassoModel = JuMP.Model(solver = GurobiSolver(TimeLimit = 30, OutputFlag = 0));
	gamma = 10

	#Define variables
	@variable(lassoModel, b[1:bCols]) #Beta values

	@variable(lassoModel, v[1:bCols]) # auxiliary variables for abs

	@variable(lassoModel, T) #First objective term
	@variable(lassoModel, G) #Second objective term

	#Define objective function (5a)
	@objective(lassoModel, Min, T+G)

	#println("Trying to implement new constraint")
	xSquareExpr = @expression(lassoModel, 0*b[1]^2)
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
		   coef += standX[j,l]^(2)
		end
		append!(xSquareExpr, @expression(lassoModel, coef*b[l]^2))
	end
	#println("Implemented x^2")

	ySquaredExpr = @expression(lassoModel, 0*b[1])
	for j = 1:nRows
		append!(ySquaredExpr,@expression(lassoModel, standY[j,1]^2))
	end
	#println("Implemented y^2")

	simpleBetaExpr = @expression(lassoModel, 0*b[1])
	for l = 1:bCols
		coef = 0
		for j = 1:nRows
			coef += -1*2*standX[j, l]*standY[j]
		end
		append!(simpleBetaExpr, @expression(lassoModel, coef*b[l]))
	end
	#println("Implemented simpleBetaExpr")

	crossBetaExpr = @expression(lassoModel, 0*b[1]*b[2])
	iter = 1
	for l = 1:bCols
		for k = (l + 1):bCols
			coef = 0
			for j = 1:nRows
				coef += 2*standX[j,l]*standX[j,k]
			end
			append!(crossBetaExpr, @expression(lassoModel, coef*b[l,1]*b[k,1]))
		end
		#println("Finished loop $l/$bCols")
	end
	#println("Implemented crossBetaExpr")
	totalExpr = @expression(lassoModel, crossBetaExpr+simpleBetaExpr+xSquareExpr+ySquaredExpr)
	@constraint(lassoModel, quadConst, totalExpr <= T)
	#println("Successfully added quadratic constraints")

	#Second objective term
	@constraint(lassoModel, 1*b .<= v) #from 2bCols to 3bCols-1
	@constraint(lassoModel, -1*b .<= v) #from 3bCols to 4bCols-1
	oneNorm = sum(v[i] for i=1:bCols)

	#gamma[g]*oneNorm <= G ---> -G <= -gamma[g]*oneNorm --> G >= gamma[g]*oneNorm
	@constraint(lassoModel, gammaConstr, 1*oneNorm <= G) #4bCols

	JuMP.build(lassoModel)

	return internalmodel(lassoModel)
end

function solveLasso(model, nGammas, Xtrain, Ytrain, Xpred, Ypred)
	nGammas = nGammas
	ISRSquared = zeros(nGammas)
	OOSRSquared = zeros(nGammas)
	gammaArray = logspace(0, 300, nGammas)
	gamma = 0
	tol = 1e-6
	bSolved = 0
	for j in 1:length(gammaArray)
		gamma = gammaArray[j]
		changeGammaLasso(model, gamma)

		Gurobi.writeproblem(model, "testproblem.lp")

		#solve problem
		status = Gurobi.optimize!(model)

		#Get solution
		sol = Gurobi.getsolution(model)

		#Get parameters
		bSolved = sol[1:bCols]

		#Shrink values to 0 if within tolerance "tol"
		for i=1:length(bSolved)
			if bSolved[i] < tol
				if bSolved[i] > -tol
					bSolved[i] = 0
				end
			end
		end
		#In-Sample R-squared value
		errors = (Ytrain-Xtrain*bSolved)
		errorTotal = sum(errors[i]^2 for i=1:length(errors))
		errorsMean = Ytrain-mean(Ytrain)
		errorMeanTotal = sum(errorsMean[i]^2 for i=1:length(errorsMean))
		ISRSquared[j] = 1-(errorTotal/errorMeanTotal)


		#Out of sample test
		oosErrors = Ypred - Xpred*bSolved
		oosErrorTotal = sum(oosErrors[i]^2 for i=1:length(oosErrors))
		oosErrorsMean = Ypred - mean(Ypred)
		oosErrorsMeanTotal = sum(oosErrorsMean[i]^2 for i=1:length(oosErrorsMean))
		OOSRSquared[j] = 1-(oosErrorTotal/oosErrorsMeanTotal)
	end
	return ISRSquared, OOSRSquared
end


testRuns = 5
trainingSize = 50
predictions = 10
nGammas = 2
ISRSquared = zeros(testRuns, nGammas)
OOSRSquared = zeros(testRuns, nGammas)

i = 5
Xtrain = allData[(1+(i-1)*trainingSize):(trainingSize+(i-1)*trainingSize), 1:bCols]
Ytrain = allData[(1+(i-1)*trainingSize):(trainingSize+(i-1)*trainingSize), bCols+1]
lassoModel = buildLasso(Xtrain,Ytrain)
Gurobi.optimize!(lassoModel)
Gurobi.writeproblem(lassoModel, "testproblem.lp")
changeGammaLasso(lassoModel, 0)
status = Gurobi.optimize!(lassoModel)
Xpred  = allData[(trainingSize+(i-1)*trainingSize):(trainingSize+(i-1)*trainingSize+predictions), 1:bCols]
Ypred  = allData[(trainingSize+(i-1)*trainingSize):(trainingSize+(i-1)*trainingSize+predictions), bCols+1]
solveLasso(lassoModel, nGammas, Xtrain, Ytrain, Xpred, Ypred)



windowSize = 50
testSize = 100
yHatPred = zeros(testSize,50)
for i = 1:testSize
	if i%25 == 0
		println("Iteration $i out of $testSize iniated")
	end
	X = allData[1+i+100:100+windowSize+i,1:bCols]
	y = allData[1+i+100:100+windowSize+i,bCols+1]
	lassoM = buildLasso(X,y)

	X_pred = allData[windowSize+i+1+100:100+windowSize+i+2,1:bCols]
	y_pred = allData[windowSize+i+1+100:100+windowSize+i+2,bCols+1]
	yHatPred[i,:] = solveLasso(lassoM,X_pred)
end
y_vali = allData[windowSize+2:windowSize+2+testSize-1]

RMSE = zeros(50)
RsquareArr = zeros(50)
for j = 1:50
	SSres = sum((y_vali[i] - yHatPred[i,j])^2 for i=1:length(y_vali))
	n=length(y)

	SSTO = sum((y_vali[i]-mean(y_vali))^2 for i=1:length(y_vali))
    Rsquared = 1-SSres/SSTO

	RsquareArr[j] = Rsquared
	RMSE[j] = sqrt(SSres/n)
end




using Convex
using SCS
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Skipper's path
path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"
#HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
#mainData = loadConcrete(path)
#fileName = path*"/Results/Concrete/Concrete"
#mainData = loadHousingData(path)
#fileName = path*"/Results/HousingData/HousingData"
#mainData = loadCPUData(path)
#fileName = path*"/Results/CPUData/CPUData"

#Reset HPC path
#path = "/zhome/9f/d/88706/SpecialeCode/Thesis/ML"
#cd(path)

mainDataArr = Array(mainData)

colNames = names(mainData)

nRows = size(mainDataArr)[1]
nCols = size(mainDataArr)[2]

mainXarr = mainDataArr[:,1:nCols-1]

# Transform #
mainXarr = expandWithTransformations(mainXarr)

# Standardize
standX = zScoreByColumn(mainXarr)
allData = hcat(standX,mainDataArr[:,nCols])

windowSize = 50
testSize = 150

nGammas = 10
gammaArray = logspace(0, 3, nGammas)
gamma = 0

yHatPred = zeros(testSize,nGammas)
# Create a (column vector) variable of size n x 1.
bCols=size(allData)[2]-1
beta = Variable(bCols)
v = Variable(bCols)
for i = 1:testSize

	if i%25 == 0
		println("Iteration $i out of $testSize iniated")
	end
	X = allData[1+i:windowSize+i,1:bCols]
	y = allData[1+i:windowSize+i,bCols+1]

	for j=1:nGammas
		gamma = gammaArray[j]
		problem = minimize(sumsquares(y - X * beta)+gamma*norm(beta,1))
		# Solve the problem by calling solve!
		solve!(problem, SCSSolver(verbose=false));
		# Check the status of the problem
		problem.status # :Optimal, :Infeasible, :Unbounded etc.
		# Get the optimal value
		problem.optval
		yHatPred[i,j] = (allData[windowSize+i+1, 1:bCols]'*beta.value)[1]
	end
end

y_vali = allData[windowSize+2:windowSize+2+testSize-1,end]

RMSE = zeros(50)
RsquareArr = zeros(50)
for j = 1:50
	SSres = sum((y_vali[i] - yHatPred[i,j])^2 for i=1:length(y_vali))
	n=length(y)

	SSTO = sum((y_vali[i]-mean(y_vali))^2 for i=1:length(y_vali))
    Rsquared = 1-SSres/SSTO

	RsquareArr[j] = Rsquared
	RMSE[j] = sqrt(SSres/n)
end

maximum(RsquareArr)
minimum(RMSE)


using Gurobi

srand(123)
println(" \n \n \n solving Convex \n \n \n")
n =
m = 100, 10
h = randn(n)

lambda = 0.01

x = full(sprandn(m, 0.05))
y = conv(h,x)+randn(n+m-1)
println(" \n \n \n solving JuMP \n \n \n")
using JuMP

A = hcat([[zeros(i);h;zeros(m-1-i)] for i = 0:m-1]...) # Full Matrix

M = Model(solver = GurobiSolver())
@variables M begin
        x[1:m]
        t[1:m]
        w
end
@objective(M,Min,0.5*w+lambda*ones(m)'*t)
@constraint(M, soc, norm( [1-w;2*(A*x-y)] ) <= 1+w)
@constraint(M,  x .<= t)
@constraint(M, -t .<= x)

solve(M)
xJuMP = getvalue(x)
norm(xcvxpy-xJuMP)




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
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"
mainData = loadIndexDataNoDur(path)
fileName = path*"/Results/IndexData/IndexData"
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



nGammas = 2
trainingSize = 12
predictions = 1
testRuns = nRows-trainingSize-predictions
ISR = zeros(nGammas, testRuns)
OOSR = zeros(nGammas, testRuns)
Indi = zeros(nGammas, testRuns)
Indi2 = zeros(nGammas, testRuns)
gammaArray = logspace(0, 1, nGammas)

for r = 1:(nRows-trainingSize-predictions)
	Xtrain = allData[r:(trainingSize+(r-1)), 1:bCols]
	Ytrain = allData[r:(trainingSize+(r-1)), bCols+1]
	Xpred  = allData[(trainingSize+(r-1)+1):(trainingSize+(r-1)+predictions), 1:bCols]
	Ypred  = allData[(trainingSize+(r-1)+1):(trainingSize+(r-1)+predictions), bCols+1]
	for g = 1:nGammas
		gamma = gammaArray[g]
		ISR[g, r], OOSR[g, r], Indi[g, r], Indi2[g, r] = solveLasso(Xtrain, Ytrain, Xpred, Ypred, gamma)
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
combinedArray = hcat(round.(gammaArray,3), ISR)
runCounter = collect(0:testRuns)'
ISRcomb = vcat(runCounter, combinedArray)
writedlm("ISRTestNoDur121IndiPortfolio.CSV", ISRcomb,",")
combinedArray = hcat(round.(gammaArray,3), OOSR)
OOSRcomb = vcat(runCounter, combinedArray)
writedlm("OOSRTestNoDur121Portfolio.CSV", OOSRcomb,",")

combinedArray = hcat(round.(gammaArray,3), Indi)
Indicomb = vcat(runCounter, combinedArray)
writedlm("IndicatorTestNoDur121IndiPortfolio.CSV", Indicomb,",")


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
