using PRONTO
using StaticArrays
using LinearAlgebra

## --------------------- helper functions --------------------- ##

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function inprod(x)
    a = x[1:3]
    b = x[4:6]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a'   a*a'+b*b']
    return P
end

## --------------------- modeling --------------------- ##
@kwdef struct LambdaSys <: Model{6,4}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


@define_f LambdaSys begin
    H0 = [-0.5 0 0;0 0 0;0 0 -0.5]
    H1 = [0 -0.5 0;-0.5 0 0;0 0 0]
    H2 = [0 -0.5im 0;0.5im 0 0;0 0 0]
    H3 = [0 0 0;0 0 -0.5;0 -0.5 0]
    H4 = [0 0 0;0 0 -0.5im;0 0.5im 0]
    mprod(-im*(H0+u[1]*H1+u[2]*H2+u[3]*H3+u[4]*H4))*x
end

@define_l LambdaSys 0.01/2*u'*I(4)*u

@define_m LambdaSys begin
    Pl = 10*I(6)
    xf = [0;0;1;0;0;0]
    1/2*(x-xf)'*Pl*(x-xf)
end

@define_Q LambdaSys kq*I(6)
@define_R LambdaSys kr*I(4)

PRONTO.Pf(θ::LambdaSys,α,μ,tf) = SMatrix{6,6,Float64}(I(6))

# must be run after any changes to model definition
resolve_model(LambdaSys)


## --------------------- run optimization --------------------- ##

θ = LambdaSys() # instantiate a new model
τ = t0,tf = 0,5 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] # initial state
μ = t->SVector{4}(0.05*sin(t)*ones(4))
η = open_loop(θ, x0, μ, τ); # guess trajectory
@time ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## --------------------- outputs --------------------- ##

using MAT

ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_lambda_5.mat","w")
write(file,"Uopt",us)
close(file)