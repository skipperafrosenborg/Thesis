using JuMP
#using CPLEX
#using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
include("DataLoad.jl")
println("Leeeeroooy Jenkins")

#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")
path = "$(homedir())/Documents/GitHub/Thesis/Data"


#Skipper's path
#path = "/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data"

#mainData = loadHousingData(path)
#mainData = loadCPUData(path)
mainData = loadElevatorData(path)

dataSize = size(mainData)
colNames = names(mainData)

#Converting it into a datamatrix instead of a dataframe
combinedData = Array(mainData)
nRows = size(combinedData)[1]
nCols = size(combinedData)[2]

### STAGE 1 ###
println("STAGE 1 INITIATED")
#Define solve
m = Model(solver = GurobiSolver())

#Add binary variables variables
@variable(m, 0 <= z[1:nCols] <= 1, Bin )

#Calculate highly correlated matix
HC = cor(combinedData)

#Define objective function
@objective(m, Max, sum(z[i] for i=1:length(z)))

#Define max correlation and constraints
rho = 0.8
for i=1:length(z)
	for j=1:length(z)
		if i != j
			if HC[i,j] >= rho
				@constraint(m,z[i]+z[j] <= 1)
			end
		end
	end
end

#Print model
#print(m)

#Get solution status
status = solve(m)

#Get objective value
println("Objective value kMax: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)

#Get solution value
#zSolved = getvalue(z)
println("STAGE 1 DONE")

### STAGE 2 ###
println("STAGE 2 INITIATED")
println("Standardizing data")

#Split data by X and y
y = combinedData[:,nCols]
X = combinedData[:,1:nCols-1]

# TRANSFORMATIONS ###
X = expandWithTransformations(X)

# Standardize
standX = zScoreByColumn(X)
standY = zscore(y)

function solveForBeta(x, y, k)
	standX = copy(x)
	X = copy(x)
	standY = copy(y)
	nRows = size(X)[1]
	#Initialise values for check later
	kValue = []
	RsquaredValue = []
	bSolved = []
	warmStartBeta = []
	warmstartZ = []
	HC = cor(X)
	for i in k:k#:5:kmax
		#println("Setup model")
		#Define parameters and model
		bCols = size(X)[2]

		stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
		gamma = 10

		consplit = 10
		#Define variables
		@variable(stage2Model, b[1:bCols])

		@variable(stage2Model, T)
		#@variable(stage2Model, T[1:consplit])
		#Define binary variable (5b)
		@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )
		#@variable(stage2Model, O)

		println("Calculating warmstart solution")
		#Calculate warmstart
		warmStartError = 1e6
		for j in 1:5
			warmStartBetaTemp = gradDecent(standX, standY, 30000, 1e-6, i, HC)
			tempError = norm(standY- standX*warmStartBetaTemp)^2
			#println("Iteration $j of 5: Warmstart error is:", tempError)
			if tempError < warmStartError
				warmStartError = copy(tempError)
				warmStartBeta = copy(warmStartBetaTemp)
				#println("Iteration $j of 5: Changed warmstartBeta")
			end
		end

		println("Warmstart error is:", warmStartError)

		#Set warmstart values
		warmstartZ = zeros(warmStartBeta)
		for j in 1:bCols
			setvalue(b[j], warmStartBeta[j])
			if isequal(warmStartBeta[j],0)
				setvalue(z[j], 0)
			else
				setvalue(z[j], 1)
				warmstartZ[j] = 1
			end
		end

		#Calculating bigM
		tau = 2
		bigM = tau*norm(warmStartBeta, Inf)

		#Define objective function (5a)
		@objective(stage2Model, Min, T)#sum(T[j] for j= 1:consplit))

		#@constraint(stage2Model, norm(standY - standX*b) <= T)




		expr = @expression(stage2Model, (standY[1]-standX[1,:]'*b)^2)
		for l=2:nRows
			tempExpr = @expression(stage2Model, (standY[l]-standX[l,:]'*b)^2)
			append!(expr, tempExpr)
			#println("Constraint $l added")
		end
		println("Successfully added quadratic constraints")
		@constraint(stage2Model, expr <= T)

		#@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j=1:nRows) <= T)

		#=
		for k = 1:(consplit-1)
				@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j = Int64(1+(k-1)*floor(nRows/consplit)):Int64(floor(nRows/consplit)+(k-1)*floor(nRows/consplit))) <=   T[k])
				println("Constraint $k added")
		end
		=#
		#@constraint(stage2Model, sum((standY[j] - standX[j,:]'*b)^2 for j = Int64(1+floor(nRows/consplit)+(consplit-1)*floor(nRows/consplit)):Int64(nRows)) <=   T[consplit])


		#@constraint(stage2Model, 1 <=T)
		#Define constraints (5c)
		@constraint(stage2Model, conBigMN, -1*b .<= bigM*z)
		@constraint(stage2Model, conBigMP,  1*b .<= bigM*z)

		#Define kmax constraint (5d)
		@constraint(stage2Model, kMaxConstr, sum(z[j] for j=1:bCols) <= kmax)

		#Constraint 5f - can only select one of a pair of highly correlated features
		rho = 0.8
		for k=1:bCols
			for j=1:bCols
				if k != j
					if HC[k,j] >= rho
						@constraint(stage2Model,z[k]+z[j] <= 1)

						#Check for errors in warmstart
						if warmstartZ[j] + warmstartZ[k] > 1
							println("Error with index $j and $k in constraint 5f")
						end
					end
				end
			end
		end

		#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
		for j=1:(nCols-1)
			@constraint(stage2Model, z[j]+z[j+(nCols-1)]+z[j+2*(nCols-1)]+z[j+3*(nCols-1)] <= 1)

			#Check for errors in warmstart
			if warmstartZ[j]+warmstartZ[j+(nCols-1)]+warmstartZ[j+2*(nCols-1)]+warmstartZ[j+3*(nCols-1)] > 1
				println("Error with index $j in constraint 5g")
			end
		end

		#Set kMax rhs constraint
		JuMP.setRHS(kMaxConstr, i)
		println("Starting to solve stage 2 model with kMax = $i")

		#print(stage2Model)

		#Solve Stage 2 model
		status = solve(stage2Model)
		println("Objective value: ", getobjectivevalue(stage2Model))

		#Get solution and calculate R^2
		bSolved = getvalue(b)
		zSolved = getvalue(z)
		Rsquared = getRSquared(standX,standY,bSolved)
		push!(kValue, i)
		push!(RsquaredValue, Rsquared)
		#println("Rsquared value is: $Rsquared for kMax = $i")

		#printNonZeroValues(bSolved)
	end
	#println("STAGE 2 DONE")
	bestRsquared = maximum(RsquaredValue)
	kBestSol = kValue[indmax(RsquaredValue)]
	println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")

	return bSolved
end

sampleSize = 50
bCols = size(X)[2]
bSample = Matrix(sampleSize, bCols)


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

function selectSampleRows(rowsWanted, nRows)
	rowsSelected = rand(1:nRows, rowsWanted)
	return rowsSelected
end

for i=1:sampleSize
	sampleRows = selectSampleRows(100, nRows)
	sampleX = createSampleX(standX, sampleRows)
	sampleY = createSampleY(standY, sampleRows)
	bSample[i,:] = solveForBeta(sampleX, sampleY, 9)
end

using Bootstrap
n_boot = 1000
samples = convert(Array{Float64, 1}, bSample[:,3])
bs1 = bootstrap(samples, mean, BasicSampling(n_boot))
bsSe = bootstrap(samples, std, BalancedSampling(n_boot))

sd1 = Float64(0.10724)
cil = 0.99
bSample = convert(Array{Float64}, bSample)
function createConfidenceIntervalArray(sampleInput, nBoot, confLevel)
	bSamples = convert(Array{Float64}, sampleInput)
	bCols = size(bSamples)[2]
	confIntArray = Matrix(2, bCols)
	for i=1:bCols
		bs = bootstrap(bSample[:,i], mean, BasicSampling(nBoot))
		cil = confLevel
		CiEst = ci(bs, BasicConfInt(cil))
		confIntArray[1,i] = CiEst[1][2] #lower
		confIntArray[2,i] = CiEst[1][3]
	end
	return confIntArray
end

confArray99 = createConfidenceIntervalArray(bSample, 1000, 0.99)
confArray95 = createConfidenceIntervalArray(bSample, 1000, 0.95)
confArray90 = createConfidenceIntervalArray(bSample, 1000, 0.90)

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

sig = zeros(72)
sig[20]
signi = testSignificance(confArray99, confArray95, confArray90, bSample[1,:])
for i=1:72
	if signi[i] > 0
		println("Parameter $i is significant with ", signi[i])
	end
end

confArray90[2,52]
bresult = bSample[1,:]
bCols = size(bresult)[1]

baaa = 5
if baaa >= 3 && baaa <= 4
	println("Hej")
end


#If
CiEst = Array(ci(bs1, BasicConfInt(cil))) #return t0, lower, upper
CiEstLow = CiEst[1][2]
CiEstUp  = CiEst[1][3]
ci(bs1, PercentileConfInt(cil)) #return t0, lower, upper
ci(bs1, NormalConfInt(cil)) #return t0, lower, upper
ci(bs1, BCaConfInt(cil)) #return t0, lower, upper
#ci(bs1, sd1, StudentConfInt(cil), 4) #return t0, lower, upper

r = randn(100)
n_boot = 1000

## basic bootstrap
bs1 = bootstrap(r, std, BasicSampling(n_boot))

## balanced bootstrap
bs2 = bootstrap(r, std, BalancedSampling(n_boot))





##BOOTSTRAP FOR SIGNIFICANCE
srand(2)

using Distributions
using Optim

N=1000
K=3




function loglike(rho,y,x)
	rhoCols = length(rho)
    beta = rho[1:rhoCols-1]
    sigma2 = exp(rho[rhoCols])+eps(Float64)
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf.(dist,residual) #Normally distributed ERRORS
    loglikelihood = sum(contributions)
    return -loglikelihood #negative log-likelihood
end


rhoGuess = copy(bSolved)
rhoGuess = vcat(rhoGuess)

rhoCols = length(rhoGuess)
function wrapLoglike(rho)
    return loglike(rho,standY,standX)
end

optimum = optimize(wrapLoglike, rhoGuess, ConjugateGradient())
MLE = optimum.minimizer
MLE[rhoCols] = exp(MLE[rhoCols])

B=100 #amount of samples
samples = zeros(B, 72)

for b=1:20
    theIndex = sample(1:nRows,nRows)
    x = standX[theIndex,:]
    y = standY[theIndex,:]
    samples[b,:] = solveForBeta(x, y, 9)
	#println("Iteration $b/$B")
end
#samples[:, rhoCols] = exp(samples[:, rhoCols])

using Bootstrap
n_boot = 1000
bs1 = bootstrap(samples[:,3], mean, BasicSampling(n_boot))
cil = 0.95
ci(bs1, BasicConfInt(cil)) #return t0, lower, upper

#=
bootstrapSE = std(samples,1)

nullDistribution = samples
pvalues = ones(rhoCols)
for i=1:rhoCols
    nullDistribution[:,i] = nullDistribution[:,i]-mean(nullDistribution[:,i])
end
nullDistribution[:, rhoCols] = 1 + nullDistribution[:, rhoCols]

pvalues = [mean(abs(MLE[i]).<abs(nullDistribution[:,i])) for i=1:rhoCols]
for i=1:rhoCols
	if pvalues[i] >= 0.05
		println("parameter $i has a p-value of ", pvalues[i])
	end
end
=#
