#Pkg.add("DataFrames") dataframes like in R to keep track of datafiles
#Pkg.add("CSV") Importing csv files
#Pkg.add("JuMP") JuMP package for writing models
#Pkg.add("Clp") Clp for a solver

using DataFrames;
using CSV;

cd("$(homedir())/Documents/GitHub/Thesis/Data")
#All on monthly data
mainData = CSV.read("Monthly - Average Value Weighted Returns.csv")
dataSize = size(mainData)
colNames = names(mainData)
nCols = dataSize[2]
nRows = dataSize[1]

stringMatrix = Array(mainData[2:1097,2:11])
dataMatrix2 = parse.(Float64, stringMatrix)

firmSizeData = CSV.read("Monthly - Average Firm Size.csv")
equalWeightData = CSV.read("Monthly - Average Equal Weighted Returns.csv")
numberOfFirms = CSV.read("Monthly - Number of Firms in Portfolios.csv")
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
using Ipopt
using GLPKMathProgInterface

m = Model(solver = GLPKSolverMIP())
@variable(m, 0 <= z[1:10] <= 1, Bin )


HC = cor(dataMatrix2)

rho = 0.8
@objective(m, Max, sum(z[i] for i=1:length(z)))
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
zSolved = getvalue(z)
length(zSolved)
for i=1:length(zSolved)
	if zSolved[i]==0
		println("z[$i] = 0")
	end
end
