module readLP

export loadLP

using CPLEX
import MathProgBase
const MPB=MathProgBase # shorthand for long module name

function loadLP(filename,solver=CplexSolver())
    println("pwd:", pwd())
    m =  MPB.LinearQuadraticModel(solver)
    MPB.loadproblem!(m,filename) # load what we actually want

    return  MPB.getconstrmatrix(m), MPB.getobj(m), MPB.getvarLB(m), MPB.getvarUB(m),
            MPB.getconstrLB(m), MPB.getconstrUB(m), MPB.getvartype(m)
end

end
