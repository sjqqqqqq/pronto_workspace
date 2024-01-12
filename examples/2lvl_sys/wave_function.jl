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

@kwdef struct WaveFunc <: PRONTO.Model{8,1}
    kl::Float64
    kr::Float64
    T::Float64 
end

@define_f WaveFunc begin
    E0 = 0.0
    E1 = 0.5725
    H0 = diagm([E0, E1])
    H00 = kron(I(2),H0)
    H1 = [0 -1im; 1im 0]
    H11 = kron(I(2),H1)
    return 2 * π * mprod(-im * (H00 + u[1]*H11)) * x
end

@define_l WaveFunc begin
    1/2*max(kl,10*100^(-0.2*t),10*100^(0.2*(t-T)))*u'*I*u
end

@define_m WaveFunc begin    # todo: change terminal cost to gate fidelity
    ψ1 = [1;0]
    ψ2 = [0;1]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(8)*(x-xf)
end

@define_Q WaveFunc I(8)
@define_R WaveFunc kr*I(1)
PRONTO.Pf(θ::WaveFunc,α,μ,tf) = SMatrix{8,8,Float64}(I(8))

resolve_model(WaveFunc)

## Compute the optimal solution

θ = WaveFunc(kl=0.2,kr=10,T=200) # 200ns optimal pulse
# θ = WaveFunc(kl=0.2,kr=10,T=600) # 600ns optimal pulse
τ = t0,tf = 0,θ.T
ψ1 = [1;0]
ψ2 = [0;1]
x0 = SVector{8}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
# μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*sin(2*π*0.5725*t))
μ = t->SVector{1}(0.001*sin(2*π*0.5725*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-6); # optimal trajectory

## Plot the results

using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "population")

lines!(ax1, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[5]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[6]^2 for t in ts], linewidth = 2, label = "|1⟩")
axislegend(ax2, position = :rc)

lines!(ax3, ts, [ξ.x(t)[3]^2+ξ.x(t)[7]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax3, ts, [ξ.x(t)[4]^2+ξ.x(t)[8]^2 for t in ts], linewidth = 2, label = "|1⟩")
axislegend(ax3, position = :rc)

display(fig)

# ## output results
# using MAT

# dt = 0.1453218
# ts = 0:dt:tf
# us = [ξ.u(t)[1] for t in ts]
# file = matopen("Uopt_Flux_X_600ns.mat", "w")
# write(file, "Uopt_X", us)
# close(file)