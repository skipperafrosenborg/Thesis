#Pkg.add("CSV");
#Pkg.add("DataFrames");

using DataFrames;
using CSV;

println("Leeeeroooy Jenkins")

#Esben's path
#cd("$(homedir())/Documents/GitHub/Thesis/Data")

#Skipper's path
cd("/Users/SkipperAfRosenborg/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data")

#All on monthly data
mainData = CSV.read("AmesHousingModClean.csv", delim = ';', nullable=false)
#mainData = CSV.read("AmesHousingMod.csv")

dataSize = size(mainData)
nRowsMain = dataSize[1]
nColsMain = dataSize[2]
colNames = names(mainData)

#Converting it into a datamatrix instead of a dataframe
mainDataMatrix = Array(mainData)

using JuMP
using CPLEX

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
length(zSolved)
for i=1:length(zSolved)
	if zSolved[i]==0
		println("z[$i] = 0")
	end
end
println("STAGE 1 DONE")

### STAGE 2 ###
println("STAGE 2 INITIATED")
println("Standardizing data")
using StatsBase

#Standardize data by coulmn
y = combinedData[1:1000,nColsMain]
X = combinedData[1:1000,1:nColsMain-1]

"""
Function that returns the zscore column zscore for each column in the Matrix.
Must have at least two columns
"""
function zScoreByColumn(X::AbstractArray{T}) where {T<:Real}
	standX = Array{Float64}(X)
	for i=1:size(X)[2]
		temp = zscore(X[:,i])
		if any(isnan(temp))
			println("Column $i remains unchanged as std = 0")
		else
			standX[:,i] = copy(zscore(X[:,i]))
		end
	end
	return standX
end

standX = zScoreByColumn(X)
standY = zscore(y)

println("Setup model")
#Define parameters and model
bCols = size(X)[2]

stage2Model = Model(solver = CplexSolver(CPX_PARAM_MIPDISPLAY = 3))
bigM = 10
gamma = 10

#Define variables
@variable(stage2Model, b[1:bCols])
@variable(stage2Model, T)
#@variable(stage2Model, O)

#Define objective function (5a)
@objective(stage2Model, Min, T)
@constraint(stage2Model, norm(standY - standX*b) <= T)

#Define binary variable (5b)
@variable(stage2Model, 0 <= z[1:bCols] <= 1, Bin )

#Define constraints (5c)

for i=1:bCols
	@constraint(stage2Model, -bigM*z[i] <= b[i])
	@constraint(stage2Model, b[i] <= bigM*z[i])
end

#Define kmax constraint (5d)
@constraint(stage2Model, kMaxConstr, sum(z[i] for i=1:bCols) <= kmax)

#=
@variable(stage2Model, v[1:bCols])
for i=1:bCols
	@constraint(stage2Model, b[i]  <= v[i])
	@constraint(stage2Model, -b[i] <= v[i])
end

@constraint(stage2Model, sum(v[i] for i=1:bCols) <= O)
=#


#print(stage2Model)
i=1
#for i in range(1,Int64(kmax))
for i in 4:5
	#JuMP.setRHS(kMaxConstr,-bCols+i)
	JuMP.setRHS(kMaxConstr, i)
	println("Starting to solve stage 2 model with kMax = $i")
	status = solve(stage2Model)
	println("Objective value: ", getobjectivevalue(stage2Model))
	bSolved = getvalue(b)
	zSolved = getvalue(z)
	for i=1:length(b)
		if !isequal(bSolved[i], 0)
			println("b[$i] = ", bSolved[i])
			println("z[$i] = ", zSolved[i])
		end
	end
end
	#println("Solve model for kMax = $i out of 195")
#end

println("STAGE 2 DONE")
SSTO = sum((standY[i]-mean(standY))^2 for i=1:length(standY))
Rsquared = 1-(getobjectivevalue(stage2Model))/SSTO
#Rsquared = (getobjectivevalue(stage2Model)-getvalue(O)*gamma)/SSTO
