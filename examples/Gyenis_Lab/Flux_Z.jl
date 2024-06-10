using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## helper functions

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function GateFidelityZ(xT)
    Z = [1 0; 0 -1]
    UT = [xT[1]+xT[5]im xT[3]+xT[7]im; xT[2]+xT[6]im xT[4]+xT[8]im]
    return (abs(tr(UT*Z'))^2+2)/6
end

## Two-level Qubit model

@kwdef struct FluxZ <: PRONTO.Model{9,1}
    kl::Float64
    kr::Float64
    T::Float64 
end

@define_f FluxZ begin
    E0 = 0.0
    E1 = 0.65225
    H0 = diagm([E0, E1])
    H00 = kron(I(2),H0)
    H1 = [0 -1im; 1im 0]
    H11 = kron(I(2),H1)
    return [2 * π * mprod(-im * (H00 + x[9]*H11)) * x[1:8];u[1]]
end

@define_l FluxZ begin
    kl/2*u'*I*u + 1/2*max(kl,10*100^(-0.2*t),10*100^(0.2*(t-T)))*x[9]'*I*x[9]
end

@define_m FluxZ begin
    ψ1 = [1;0]
    ψ2 = [0;-1]
    xf = vec([ψ1;ψ2;0*ψ1;0*ψ2;0])
    return 1/2*(x-xf)'*I(9)*(x-xf)
end

@define_Q FluxZ I(9)
@define_R FluxZ kr*I(1)
PRONTO.Pf(θ::FluxZ,α,μ,tf) = SMatrix{9,9,Float64}(I(9))

resolve_model(FluxZ)

## Compute the optimal solution

θ = FluxZ(kl=0.03,kr=10,T=100)
τ = t0,tf = 0,θ.T
ψ1 = [1;0]
ψ2 = [0;1]
x0 = SVector{9}(vec([ψ1;ψ2;0*ψ1;0*ψ2;0]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*sin(2*π*0.65225*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory