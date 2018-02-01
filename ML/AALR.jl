#Pkg.add("DataFrames") dataframes like in R to keep track of datafiles
#Pkg.add("CSV") Importing csv files
#Pkg.add("JuMP") JuMP package for writing models
#Pkg.add("Clp") Clp for a solver

using DataFrames;
using CSV;
println("Leeeeroooy Jenkins")

cd("$(homedir())/Documents/GitHub/Thesis/Data")
#All on monthly data
mainData = CSV.read("Monthly - Average Value Weighted Returns.csv")
dataSize = size(mainData)
nColsMain = dataSize[2]
nRowsMain = dataSize[1]
colNames = names(mainData)

#Converting it into a datamatrix instead of a dataframe
stringMatrix = Array(mainData[2:nRowsMain,2:nColsMain])
dataMatrixMain = parse.(Float64, stringMatrix)

#Importing exogenous factors
firmSizeData = CSV.read("Monthly - Average Firm Size.csv")
stringMatrix = Array(firmSizeData[2:nRowsMain,2:nColsMain])
dataMatrixFirmSize = parse.(Float64, stringMatrix)

equalWeightData = CSV.read("Monthly - Average Equal Weighted Returns.csv")
numberOfFirms = CSV.read("Monthly - Number of Firms in Portfolios.csv")
stringMatrix = Array(numberOfFirms[2:nRowsMain,2:nColsMain])
dataMatrixFirms = parse.(Float64, stringMatrix)
println("Monthly Data is loaded")

##### Making a model
#Tips
#=
@variable(m, x )              # No bounds
@variable(m, x >= lb )        # Lower bound only (note: 'lb <= x' is not valid)
@variable(m, x <= ub )        # Upper bound only
@variable(m, lb <= x <= ub )  # Lower and upper bounds
=#
#https://jump.readthedocs.io/en/latest/quickstart.html#creating-a-model

using JuMP
using CPLEX

combinedData = hcat(dataMatrixMain, dataMatrixFirmSize, dataMatrixFirms)
nCols = size(combinedData)[2]
nRows = size(combinedData)[1]

m = Model(solver = CplexSolver())
@variable(m, 0 <= z[1:nCols] <= 1, Bin )


HC = cor(combinedData)

@objective(m, Max, sum(z[i] for i=1:length(z)))

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
#@constraint(m, z[2] + z[3] <= 1)

print(m)

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
kmax = getobjectivevalue(m)
zSolved = getvalue(z)
length(zSolved)
for i=1:length(zSolved)
	if zSolved[i]==0
		println("z[$i] = 0")
	end
end


#STAGE 2
using StatsBase
standData = zscore(dataMatrixMain)
standCombinedData = zscore(combinedData)

#Want to make sure that the previous informations gets used to estimate
#current values (So t=9 information should predict t=10 observation)
#Have to move combinedData matrix around, so observation 1097 has combinedData row 1096


standDataCopy = copy(standData[2:nRows,1:nColsMain-1])
standCombDataCopy = copy(standCombinedData[1:(nRows-1),1:nCols])

z = 0
stage2Model = Model(solver = CplexSolver())
bigM = 100
gamma = 0.01
@variable(stage2Model, 0 <= z[1:nCols] <= 1, Bin )
@variable(stage2Model, b[1:nCols])
@variable(stage2Model, T)
@variable(stage2Model, O)


@objective(stage2Model, Min, T + gamma*O)
@constraint(stage2Model, norm(standDataCopy[:,1] - standCombDataCopy*b) <= T)

for i=1:nCols
	@constraint(stage2Model, -bigM*z[i] <= b[i])
	@constraint(stage2Model, b[i] <= bigM*z[i])
end

@constraint(stage2Model, sum(z[i] for i=1:nCols) <= kmax)

@variable(stage2Model, v[1:nCols])
for i=1:nCols
	@constraint(stage2Model, b[i]  <= v[i])
	@constraint(stage2Model, -b[i] <= v[i])
end

@constraint(stage2Model, sum(v[i] for i=1:nCols) <= O)



#print(stage2Model)

status = solve(stage2Model)

println("Objective value: ", getobjectivevalue(stage2Model))
bSolved = getvalue(b)
for i=1:length(b)
	if b[i]!=0
		println("b[$i] = ", bSolved[i])
	end
end

SSTO = sum((standDataCopy[i,1]-mean(standDataCopy[:,1]))^2 for i=1:nRows-1)
Rsquared = 1-(getobjectivevalue(stage2Model)/SSTO)
