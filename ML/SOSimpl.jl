#Pkg.add("CSV");
#Pkg.add("DataFrames");
using JuMP
using CPLEX
using ConditionalJuMP
using Gurobi
using StatsBase
using DataFrames
using CSV
include("SupportFunction.jl")
println("Leeeeroooy Jenkins")

using Bootstrap
r = randn(100)

n_boot = 1000

## basic bootstrap
bs1 = bootstrap(r, std, ResidualSampling(n_boot))

## balanced bootstrap
bs2 = bootstrap(r, std, BalancedSampling(n_boot))

bias(bs1)
se(bs1)

## calculate 95% confidence intervals
cil = 0.95;

## basic CI
bci1 = ci(bs1, BasicConfInt(cil));

## percentile CI
bci2 = ci(bs1, PercentileConfInt(cil));

## BCa CI
bci3 = ci(bs1, BCaConfInt(cil));

## Normal CI
bci4 = ci(bs1, NormalConfInt(cil));


#Esben's path
cd("$(homedir())/Documents/GitHub/Thesis/Data")

#Skipper's path
#cd("/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data")

#All on monthly data
mainData = CSV.read("AmesHousingModClean.csv", delim = ';', nullable=false)
#mainData = CSV.read("machine.data", header=["vendor name","Model name","MYCT",
#	"MMIN","MMAX","CACH","CHMIN","CHMAX","PRP","ERP"], datarow=1, nullable=false)
#mainData = copy(mainData[:,3:9])

dataSize = size(mainData)
nRowsMain = dataSize[1]
nColsMain = dataSize[2]
colNames = names(mainData)

#Converting it into a datamatrix instead of a dataframe
mainDataMatrix = Array(mainData)

combinedData = copy(mainDataMatrix)
nRows = size(combinedData)[1]
nCols = size(combinedData)[2]

### STAGE 1 ###
println("STAGE 1 INITIATED")
#Define solve
m = Model(solver = CplexSolver())

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
println("Objective value: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)

#Get solution value
zSolved = getvalue(z)
println("STAGE 1 DONE")

### STAGE 2 ###
println("STAGE 2 INITIATED")
println("Standardizing data")

#Standardize data by coulmn
y = combinedData[:,nColsMain]
X = combinedData[:,1:nColsMain-1]

### TRANSFORMATIONS ###
X = combinedData[:,1:nColsMain-1]
X = expandWithTransformations(X)

standX = zScoreByColumn(X)
standY = zscore(y)
using Lasso
using GLMNet
LassoModel = fit(LassoPath, standX, standY, Normal(), standardize = false, cd_maxiter = 2000000)

using GLM
glm(@formula(standY ~ standX), )
println("DOOOOOOOOOOOOOOOOOOOOOONE")
#for i in 1:30
#	warmStartBeta = gradDecent(standX, standY, 20000, 1e-6, 20)
   	#println("Iteration $i:\t R^2 is:", getRSquared(standX, standY, warmStartBeta))
#end

kValue = []
RsquaredValue = []
HC = cor(X)
for i in 1:2:kmax
	println("Setup model")
	#Define parameters and model
	bCols = size(X)[2]

	#stage2Model = Model(solver = CplexSolver(CPX_PARAM_TILIM = 400))
	stage2Model = Model(solver = GurobiSolver(Presolve=-1, TimeLimit = 200))
	gamma = 10

	#Define variables
	@variable(stage2Model, b[1:bCols])
	@variable(stage2Model, T)
	#@variable(stage2Model, O)

	println("Calculating warmstart solution")
	#Calculate warmstart
	warmstartError = 1e6
	warmStartBeta = []
	for i in 1:20
		warmStartBetaTemp = gradDecent(standX, standY, 20000, 1e-6, i)
		tempError = norm(standY- standX*warmStartBetaTemp)
		println("Warmstart error is:", norm(standY- standX*warmStartBetaTemp))
		if tempError < warmstartError
			warmstartError = tempError
			warmStartBeta = warmStartBetaTemp

	end

	for j in 1:bCols
		setvalue(b[j], warmStartBeta[j])
	end

	#Calculating bigM
	tau = 2
	bigM = tau*norm(warmStartBeta, Inf)

	#Define objective function (5a)
	@objective(stage2Model, Min, T)
	@constraint(stage2Model, norm(standY - standX*b) <= T)

	#Define binary variable (5b)
	@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

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
				end
			end
		end
	end

	#Constraint (5g) - only one transformation allowed (x, x^2, log(x) or sqrt(x))
	for j=1:(nColsMain-1)
		@constraint(stage2Model, z[j]+z[j+(nColsMain-1)]+z[j+2*(nColsMain-1)]+z[j+3*(nColsMain-1)] <= 1)
	end

	#=
	@variable(stage2Model, v[1:bCols])
	for i=1:bCols
		@constraint(stage2Model, b[i]  <= v[i])
		@constraint(stage2Model, -b[i] <= v[i])
	end

	@constraint(stage2Model, sum(v[i] for i=1:bCols) <= O)
	=#

	#Set kMax rhs constraint
	JuMP.setRHS(kMaxConstr, i)
	println("Starting to solve stage 2 model with kMax = $i")

	#print(stage2Model)

	warmstart!(stage2Model)
	println("Warmstart succes!")

	#Solve Stage 2 model
	status = solve(stage2Model)
	println("Objective value: ", getobjectivevalue(stage2Model))

	#Get solution and calculate R^2
	bSolved = getvalue(b)
	zSolved = getvalue(z)
	Rsquared = getRSquared(standX,standY,bSolved)
	push!(kValue, i)
	push!(RsquaredValue, Rsquared)
	println("Rsquared value is: $Rsquared")

	#printNonZeroValues(bSolved)
end
println("STAGE 2 DONE")
bestRsquared = maximum(RsquaredValue)
kBestSol = kValue[indmax(RsquaredValue)]
println("Bets solution found is: R^2 = $bestRsquared, k = $kBestSol")
