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

## Three-level Qubit model

@kwdef struct Flux3Ex <: PRONTO.Model{13,1}
    kl::Float64 = 0.01
    kq::Float64 = 0.0
    T::Float64 
end

@define_f Flux3Ex begin
    E0 = 0.0
    E1 = 0.5725
    E2 = 4.017875
    H0 = diagm([E0, E1, E2])
    H00 = kron(I(2),H0)
    H1 = [0 -0.15466im 0; 0.15466im 0 -0.54512im; 0 0.54512im 0]
    H11 = kron(I(2),H1)
    return [2 * π * mprod(-im * (H00 + x[13]*H11)) * x[1:12];u[1]]
end

@define_l Flux3Ex begin
    kl/2*u'*I*u + kq/2*x[1:12]'*mprod(diagm([0,0,1,0,0,1]))*x[1:12] + 1/2*max(kl,100*10^(-0.4*t),100*10^(0.4*(t-T)))*x[13]'*I*x[13]
end

@define_m Flux3Ex begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1;0])
    return 1/2*(x-xf)'*I(13)*(x-xf)
end

@define_Q Flux3Ex I(13)
@define_R Flux3Ex I(1)
PRONTO.Pf(θ::Flux3Ex,α,μ,tf) = SMatrix{13,13,Float64}(I(13))

resolve_model(Flux3Ex)

## Compute the optimal solution

θ = Flux3Ex(kl=0.1,kq=0.05,T=100)
τ = t0,tf = 0,θ.T
ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{13}(vec([ψ1;ψ2;0*ψ1;0*ψ2;0]))
# μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*0.5725*t))
μ = t->SVector{1}(0.02*sin(2*π*0.5725*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## Plot the results

using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "population")

lines!(ax1, ts, [ξ.x(t)[13] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[7]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[8]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax2, ts, [ξ.x(t)[3]^2+ξ.x(t)[9]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax2, position = :rc)

lines!(ax3, ts, [ξ.x(t)[4]^2+ξ.x(t)[10]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax3, ts, [ξ.x(t)[5]^2+ξ.x(t)[11]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax3, ts, [ξ.x(t)[6]^2+ξ.x(t)[12]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax3, position = :rc)

display(fig)

##
function GateFidelityX(xT)
    X = [0 1; 1 0]
    UT = [xT[1]+xT[7]im xT[4]+xT[10]im; xT[2]+xT[8]im xT[5]+xT[11]im]
    return (abs(tr(UT*X'))^2+2)/6
end