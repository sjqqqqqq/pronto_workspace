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

## Two-level Qubit model

@kwdef struct Flux2Ex <: PRONTO.Model{9,1}
    kl::Float64
    kr::Float64
    T::Float64 
end

@define_f Flux2Ex begin
    E0 = 0.0
    # E1 = 0.5725
    E1 = 0.65225
    H0 = diagm([E0, E1])
    H00 = kron(I(2),H0)
    H1 = [0 -1im; 1im 0]
    H11 = kron(I(2),H1)
    return [2 * π * mprod(-im * (H00 + x[9]*H11)) * x[1:8];u[1]]
end

@define_l Flux2Ex begin
    kl/2*u'*I*u + 1/2*max(kl,10*100^(-0.2*t),10*100^(0.2*(t-T)))*x[9]'*I*x[9]
end

@define_m Flux2Ex begin
    ψ1 = [1;0]
    ψ2 = [0;1]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1;0])
    return 1/2*(x-xf)'*I(9)*(x-xf)
end

@define_Q Flux2Ex I(9)
@define_R Flux2Ex kr*I(1)
PRONTO.Pf(θ::Flux2Ex,α,μ,tf) = SMatrix{9,9,Float64}(I(9))

resolve_model(Flux2Ex)

## Compute the optimal solution

θ = Flux2Ex(kl=0.03,kr=10,T=100)
# θ = Flux2Ex(kl=0.2,kr=10,T=600)
τ = t0,tf = 0,θ.T
ψ1 = [1;0]
ψ2 = [0;1]
x0 = SVector{9}(vec([ψ1;ψ2;0*ψ1;0*ψ2;0]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*sin(2*π*0.65225*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## Plot the results

using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "population")

lines!(ax1, ts, [ξ.x(t)[9] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[5]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[6]^2 for t in ts], linewidth = 2, label = "|1⟩")
axislegend(ax2, position = :rc)

lines!(ax3, ts, [ξ.x(t)[3]^2+ξ.x(t)[7]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax3, ts, [ξ.x(t)[4]^2+ξ.x(t)[8]^2 for t in ts], linewidth = 2, label = "|1⟩")
axislegend(ax3, position = :rc)

display(fig)

## output results
using MAT

dt = 0.1453218
ts = 0:dt:tf
us = [ξ.x(t)[end] for t in ts]
file = matopen("Uopt_Flux_X_200ns.mat", "w")
write(file, "Uopt_X", us)
close(file)