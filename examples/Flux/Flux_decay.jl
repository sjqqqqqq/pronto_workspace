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

@kwdef struct Decay <: PRONTO.Model{8,1}
    kl::Float64 # stage cost gain
    T1::Float64 = 1000 # T1 time
end


@define_f Decay begin
    
    E0 = 0.0
    E1 = 0.74156
    E2 = 4.017875
    H0 = 2*π*diagm([E0, E1, E2])

    H1 = 2*π*[0 -0.15466im 0; 0.15466im 0 -0.54512im; 0 0.54512im 0]

    Hc = Any[]
    push!(Hc, H1)
   
    ψ0 = [1;0;0]
    ψ1 = [0;1;0]
    ψ2 = [0;0;1]    

    L1 = 1.0 * sqrt(1/T1) * ψ0 * ψ1'
    
    # L = Matrix{ComplexF64}[]
    L = Matrix{Any}[]
    push!(L, L1)

    q_model(3,H0,Hc,L,x,u)
 
end

## ----------------------------------- compute the optimal solution ----------------------------------- ##

θ = Decay(kl=0.01,T1=1000.0)
t0,tf = τ = (0,2000)
x0 = SVector{8}(psi2x([0;1;0]))
μ = t->SVector{1}(0.0*cos(t))
η = open_loop(θ,x0,μ,τ)

## ----------------------------------- plot the results ----------------------------------- ##

using GLMakie
GLMakie.activate!()

ts = range(t0,tf,length=1001);

ρ = x2rho.(η.x(t) for t in ts)

p1 = zeros(length(ts))
p2 = zeros(length(ts))
p3 = zeros(length(ts))
p4 = zeros(length(ts))

for i in 1:length(ts)
    p1[i] = real([1,0,0]'*ρ[i]*[1,0,0])[1]
    p2[i] = real([0,1,0]'*ρ[i]*[0,1,0])[1]
    p3[i] = real([0,0,1]'*ρ[i]*[0,0,1])[1]
    p4[i] = 1/exp(1)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "time (ns)", ylabel = "population", title = "decay")
lines!(ax, ts, real(p1); color=:blue, linewidth=2, label = "|0⟩")
lines!(ax, ts, real(p2); color=:red, linewidth=2, label = "|1⟩")
lines!(ax, ts, real(p3); color=:green, linewidth=2, label = "|2⟩")
lines!(ax, ts, real(p4); color=:purple,linestyle=:dash, linewidth=2, label = "1/e")
axislegend(ax, position = :rc)

display(fig)