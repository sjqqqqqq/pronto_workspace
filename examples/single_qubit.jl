using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
using GellMannMatrices

## ----------------------------------- helper function ----------------------------------- ##

function qubit_model(H0,Hc,L,x,u)
    n = size(H0)[1]

    A = gellmann(n)
    H = H0 + u[1]*Hc
    ρ = 1/n*I(n)
    for i in 1:n^2-1
        ρ += 1/2*(x[i]*A[i])
    end
    ρ̇ = -im*(H*ρ-ρ*H) + L*ρ*L' - 1/2*(L'*L*ρ + ρ*L'*L)
    ẋ = []
    for i in 1:n^2-1
        ẋ = [ẋ; real(tr(ρ̇*A[i]))]
    end
    return ẋ
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct SQubit <: PRONTO.Model{3,1}
    kl::Float64 # stage cost gain
end

@define_f SQubit begin
    
    H0 = diagm([0,2])
    Hc = [0 1;1 0]
    L = sqrt(1/500)*[0 1;0 0] # need sqrt for decay
    # L = 1/2*[1 0;0 -1] # no need sqrt for dephase
    # L = zeros(2,2)

    qubit_model(H0,Hc,L,x,u)
 
end

@define_l SQubit begin
    kl/2*u'*I*u 
end

@define_m SQubit begin
    xf = [0,0,-1]
    1/2*(x-xf)'*(x-xf)
end

@define_Q SQubit I(3)

@define_R SQubit I(1)

resolve_model(SQubit)

PRONTO.Pf(θ::SQubit,α,μ,tf) = SMatrix{3,3,Float64}(I(3))

## ----------------------------------- solve the problem ----------------------------------- ##

θ = SQubit(kl=0.01)
t0,tf = τ = (0,2000)
x0 = SVector{3}([0,0,-1])
# x0 = SVector{3}([1,0,0])
μ = t->SVector{1}(0.0*sin(t))
η = open_loop(θ,x0,μ,τ)
# ξ,data = pronto(θ,x0,η,τ;tol=1e-4,maxiters=50);

## ----------------------------------- plot the results ----------------------------------- ##

using GLMakie
GLMakie.activate!()

ts = range(t0,tf,length=10001);

X = [0 1;1 0]
Y = [0 -im;im 0]
Z = [1 0;0 -1]

ρ = [1/2*(I(2) + η.x(t)[1]*X + η.x(t)[2]*Y + η.x(t)[3]*Z) for t in ts]
# ρ = [1/2*(I(2) + ξ.x(t)[1]*X + ξ.x(t)[2]*Y + ξ.x(t)[3]*Z) for t in ts]

p1 = zeros(length(ts))
p2 = zeros(length(ts))
p3 = zeros(length(ts))

for i in 1:length(ts)
    p1[i] = real([1 0]*ρ[i]*[1;0])[1]
    p2[i] = real([0 1]*ρ[i]*[0;1])[1]
    # p3[i] = 1/2-1/2*1/exp(1)
    p3[i] = 1 * 1/exp(1)
end

 


fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, ts, real(p1); color=:red, linewidth=2, label = "|0⟩")
lines!(ax, ts, real(p2); color=:blue, linewidth=2, label = "|1⟩")
lines!(ax, ts, real(p3); color=:green, linewidth=2, label = "1/e")
axislegend(ax, position = :rc)

display(fig)

## ----------------------------------- output the results ----------------------------------- ##
using MAT

ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_2lvl_10T.mat", "w")
write(file, "Uopt", us)
close(file)