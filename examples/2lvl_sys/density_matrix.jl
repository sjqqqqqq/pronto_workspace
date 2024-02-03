using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
using GellMannMatrices

## ----------------------------------- define helper functions ----------------------------------- ##

function x2rho(x)
    n = Int(sqrt(length(x)+1))
    A = gellmann(n; normalize=true)
    ρ = 1/n*I(n)
    for i in eachindex(A)
        ρ += 1/2*(x[i]*A[i])
    end
    return ρ
end

function psi2x(ψ)
    n = length(ψ)
    A = gellmann(n; normalize=true)
    x = Float64[]
    for i in eachindex(A)
        push!(x, real(ψ'*A[i]*ψ))
    end
    return x
end

function q_model(n,H0,Hc,L,x,u)
  
    A = gellmann(n; normalize=true)

    H = H0
    for i in eachindex(Hc)
        H += u[i]*Hc[i]
    end
    
    ρ = x2rho(x)

    ρ̇ = -im*(H*ρ-ρ*H)
    for i in eachindex(L)
        ρ̇ += L[i]*ρ*L[i]' - 1/2*(L[i]'*L[i]*ρ + ρ*L[i]'*L[i])
    end
    
    ẋ = []
    for i in eachindex(A)
        ẋ = [ẋ; real(tr(ρ̇*A[i]))]
    end
    return ẋ
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct LQubit <: PRONTO.Model{3,1}
    kl::Float64 # stage cost gain
    T1::Float64 # decay time
    T2::Float64 # dephasing time
end

@define_f LQubit begin
    
    E0 = 0.0
    E1 = 0.5725
    H0 = 2*π*diagm([E0, E1])

    H1 = 2*π*[0 -1im; 1im 0]

    Hc = Matrix{ComplexF64}[]
    push!(Hc, H1)
   
    ψ0 = [1;0]
    ψ1 = [0;1]    
    L1 = 1.0 * sqrt(1/T1) * ψ0 * ψ1'
    L2 = 1.0 * sqrt(2*1/T2) * [0 0; 0 1]

    L = Matrix{Any}[]
    push!(L, L1, L2)

    q_model(2,H0,Hc,L,x,u)
 
end

@define_l LQubit begin
    kl/2*u'*I*u 
end

@define_m LQubit begin
    X = [0 1; 1 0]
    ρf = X*x2rho(psi2x([1,0]))*X'
    real(tr((x2rho(x)-ρf)'*(x2rho(x)-ρf)))
end

@define_Q LQubit I(3)

@define_R LQubit I(1)

resolve_model(LQubit)

PRONTO.Pf(θ::LQubit,α,μ,tf) = SMatrix{3,3,Float64}(I(3))

## ----------------------------------- compute the optimal solution ----------------------------------- ##

θ = LQubit(kl=0.01,T1=1000,T2=500)
t0,tf = τ = (0,100)
x0 = SVector{3}(psi2x([1;0]))
μ = t->SVector{1}(0.001*sin(2*π*0.5725*t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-5,maxiters=50);

## ----------------------------------- plot the results ----------------------------------- ##
using GLMakie
GLMakie.activate!()

ts = range(t0,tf,length=1001);

ρ = x2rho.(η.x(t) for t in ts)

p1 = zeros(length(ts))
p2 = zeros(length(ts))

ψ0 = [1,0]
ψ1 = [0;1]

for i in 1:length(ts)
    p1[i] = real(ψ0'*ρ[i]*ψ0)[1]
    p2[i] = real(ψ1'*ρ[i]*ψ1)[1]
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, ts, real(p1); color=:red, linewidth=2, label = "|0⟩")
lines!(ax, ts, real(p2); color=:blue, linewidth=2, label = "|1⟩")
axislegend(ax, position = :rc)

display(fig)