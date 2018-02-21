using JuMP
using Gurobi

const GUR = Gurobi

println("***** Problem 1 *****")
m = Model(solver = GurobiSolver())
@variable(m, 0 <= x <= 2 )
@variable(m, 0 <= y <= 30 )
@variable(m, 0 <= z <= 30 )

@objective(m, Max, 5x + 3*y )
@constraint(m,kMaxConstr, 1x + 5y <= 3.0*z )
@constraint(m,kMaxConstr, 4x + 3y <= 1.0*z )
JuMP.setRHS(kMaxConstr, 5)
@constraint(m,kMaxConstr, 4x^2 + 3y <= 1.0*z )

print(m)

JuMP.build(m)

m2 = internalmodel(m)

println(m2)

Gurobi.setquadobj!(m2,[1.0],[1.0],[0.0])
Gurobi.updatemodel!(m2)
Gurobi.getobj(m2)



GUR.optimize!(m2)

#GUR.changecoeffs!(m2,[1,2],[1,2],[100,240])

GUR.updatemodel!(m2)

print(GUR.getconstrmatrix(m2))

println(GUR.getconstrUB(m2))

GUR.getq(m2)

GUR.optimize!(m2)

function getq(model::Model)
    nz = get_intattr(model, "NumQNZs")
    rowidx = Array{Cint}(nz)
    colidx = Array{Cint}(nz)
    val = Array{Float64}(nz)
    nzout = Array{Cint}(1)

    ret = Gurobi.@grb_ccall(getq, Cint, (
        Ptr{Void},  # model
        Ptr{Cint},  # numqnzP
        Ptr{Cint},  # qrow
        Ptr{Cint},  # qcol
        Ptr{Float64}# qval
        ),
        model,nzout,rowidx,colidx,val)

    if ret != 0
        throw(GurobiError(model.env, ret))
    end

    return rowidx, colidx, val
end

println(Gurobi.getq(m2.inner))
Gurobi.writeproblem(m2, "testproblem.lp")
