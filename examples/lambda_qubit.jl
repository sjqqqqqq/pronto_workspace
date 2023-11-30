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

@kwdef struct LQubit <: PRONTO.Model{8,4}
    kl::Float64 # stage cost gain
    kq::Float64 
end

@define_f LQubit begin
    
    H0 = [-0.5 0 0;0 0 0;0 0 -0.5]

    H1 = [0 -0.5 0;-0.5 0 0;0 0 0]
    H2 = [0 -0.5im 0;0.5im 0 0;0 0 0]
    H3 = [0 0 0;0 0 -0.5;0 -0.5 0]
    H4 = [0 0 0;0 0 -0.5im;0 0.5im 0]

    Hc = Matrix{ComplexF64}[]
    push!(Hc, H1, H2, H3, H4)
   
    
    L1 = 0.0 * [1,0,0] * [0,1,0]'
    L2 = 0.0 * [0,1,0] * [0,0,1]'
    L3 = 0.0 * [0,0,1] * [1,0,0]'

    L = Matrix{ComplexF64}[]
    push!(L, L1, L2, L3)

    q_model(3,H0,Hc,L,x,u)
 
end

@define_l LQubit begin
    kl/2*u'*I*u + kq/2*(1/3 -x[3]/2 + x[8]/(2*sqrt(3)))^2
end

@define_m LQubit begin
    xf = psi2x([0;0;1])
    1/2*(x-real(xf))'*(x-real(xf))
end

@define_Q LQubit I(8)

@define_R LQubit I(4)

resolve_model(LQubit)

PRONTO.Pf(θ::LQubit,α,μ,tf) = SMatrix{8,8,Float64}(I(8))

## ----------------------------------- compute the optimal solution ----------------------------------- ##

θ = LQubit(kl=0.01,kq=0.0)
t0,tf = τ = (0,5)
x0 = SVector{8}(psi2x([1;0;0]))
μ = t->0.3*sin(t)*ones(SVector{4})
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-4,maxiters=20);


## ----------------------------------- plot the results ----------------------------------- ##
using GLMakie
GLMakie.activate!()

ts = range(t0,tf,length=1001);

ρ = x2rho.(ξ.x(t) for t in ts)

p1 = zeros(length(ts))
p2 = zeros(length(ts))
p3 = zeros(length(ts))

for i in 1:length(ts)
    p1[i] = real([1 0 0]*ρ[i]*[1;0;0])[1]
    p2[i] = real([0 1 0]*ρ[i]*[0;1;0])[1]
    p3[i] = real([0 0 1]*ρ[i]*[0;0;1])[1]
end

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, ts, real(p1); color=:red, linewidth=2, label = "|0⟩")
lines!(ax, ts, real(p2); color=:blue, linewidth=2, label = "|1⟩")
lines!(ax, ts, real(p3); color=:green, linewidth=2, label = "|2⟩")
axislegend(ax, position = :rc)

display(fig)