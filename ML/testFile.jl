using JuMP
using Gurobi
##### Simple LP
# Example taken directly from https://jump.readthedocs.org/en/latest/quickstart.html

println("***** Problem 1 *****")
m = Model(solver = GurobiSolver())
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )
@variable(m, 0 <= z <= 30 )

@objective(m, Max, 5x + 3*y )
@constraint(m,kMaxConstr, 1x + 5y <= 3.0*z )
JuMP.setRHS(kMaxConstr, 5)

print(m)

status = solve(m)

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))
println("y = ", getvalue(z))
