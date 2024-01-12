using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
using GellMannMatrices

## ----------------------------------- constants ----------------------------------- ##

X=SMatrix{2,2,Float64}([0 1; 1 0]); # X gate matrix
Y=SMatrix{2,2,ComplexF64}([0 -im; im 0]); # Y gate matrix
Z=SMatrix{2,2,Float64}([1 0; 0 -1]); # Z gate matrix
H=1/sqrt(2)*SMatrix{2,2,Float64}([1 1; 1 -1]); # H gate matrix
S=SMatrix{2,2,ComplexF64}([1 0; 0 im]); # S gate matrix
T=SMatrix{2,2,ComplexF64}([1 0; 0 exp(im*π/4)]); # T gate matrix

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

function gate2ψ02(n,U,Q0,Q1) # Q0 and Q1 are the chosen qubit in the 3 level system 
    ψ0 = zeros(n)
    ψ0[Q0] = 1
    ψ1 = zeros(n)
    ψ1[Q1] = 1
    ψ01 = ψ0
    ψ02 = 1/sqrt(2)*(ψ0 + ψ1)
    # n = length(ψ01)
    # A = gellmann(n; normalize=true)
    W = zeros((n,n))
    W[Q0,Q0] = U[1,1]
    W[Q1,Q0]=U[2,1]
    W[Q0,Q1]=U[1,2]
    W[Q1,Q1]=U[2,2]
    ψf1 = W * ψ01
    ψf2 = W * ψ02

    return [psi2x(ψ01);psi2x(ψ02)], [psi2x(ψf1);psi2x(ψf2)]

    # if Q0 == 1 && Q1 == 2
    #     ψ02 = 1/sqrt(2) * [1,1,0]
    #     W[Q0:Q1, Q0:Q1] = U
    # elseif Q0 == 1 && Q1 == 3
    #     ψ02 = 1/sqrt(2) * [1,0,1]
    #     W[Q0,Q0]=U[1,1]
    #     W[Q1,Q0]=U[2,1]
    #     W[Q0,Q1]=U[1,2]
    #     W[Q1,Q1]=U[2,2]
    # elseif Q0 == 2 && Q1 == 3
    #     ψ02 = 1/sqrt(2) * [0,1,1]
    #     W[Q0:Q1, Q0:Q1] = U
    # end
    # U=W
    # return U, ψ02

end

function xf4gate(U,ψ0)
    n = length(ψ0)
    A = gellmann(n; normalize=true)
    xf=[]
    for i in eachindex(A)
        push!(xf,real(ψ0'*U'*A[i]*U*ψ0))
    end
    return xf
end


function q_model(H0,Hc,L,x,u)
    n = size(H0)[1]
    A = gellmann(n; normalize=true)

    H = H0
    for i in eachindex(Hc)
        H += u[i]*Hc[i]
    end
    
    ρ1 = x2rho(x[1:8])
    ρ2 = x2rho(x[9:16])

    ρ̇1 = -im*(H*ρ1-ρ1*H)
    ρ̇2 = -im*(H*ρ2-ρ2*H)
    for i in eachindex(L)
        ρ̇1 += L[i]*ρ1*L[i]' - 1/2*(L[i]'*L[i]*ρ1 + ρ1*L[i]'*L[i])
        ρ̇2 += L[i]*ρ2*L[i]' - 1/2*(L[i]'*L[i]*ρ2 + ρ2*L[i]'*L[i])
    end
    
    ẋ1 = []
    ẋ2 = []
    for i in eachindex(A)
        ẋ1 = push!(ẋ1, real(tr(ρ̇1*A[i])))
        ẋ2 = push!(ẋ2, real(tr(ρ̇2*A[i]))) 
    end
    return [ẋ1; ẋ2]
end

## ----------------------------------- define the model ----------------------------------- ##
n = 3

@kwdef struct LQubit2 <: PRONTO.Model{16,4}
    kl::Float64 # stage cost gain
    # U::SMatrix{n,n,ComplexF64} = H
    # Q0::Int64 = 1
    # Q1::Int64 = 2
end

@define_f LQubit2 begin
    
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

    q_model(H0,Hc,L,x,u)
 
end

@define_l LQubit2 begin
    kl/2*u'*I*u 
end

#ψ01 = [1,0,0]
#ψ02 = 1/sqrt(2) * [1,0,1] 
#H = [sqrt(2)/2 0 sqrt(2)/2;0 0 0;sqrt(2)/2 0 -sqrt(2)/2]

@define_m LQubit2 begin

    xf = gate2ψ02(n,H,1,2)[2]
    1/2*(x-real(xf))'*(x-real(xf))
end

@define_Q LQubit2 I(16)

@define_R LQubit2 I(4)

resolve_model(LQubit2)

PRONTO.Pf(θ::LQubit2,α,μ,tf) = SMatrix{16,16,Float64}(I(16))

## ----------------------------------- compute the optimal solution ----------------------------------- ##

θ = LQubit2(kl=0.01)#, Q0=1, Q1=2, (U, ψ02=gate2ψ02(H,1,2)))
t0,tf = τ = (0,5)
x0 = SVector{16}(gate2ψ02(n,H,1,2)[1])
μ = t->0.3*sin(t)*ones(SVector{4})
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1.2e-4,maxiters=50);


## ----------------------------------- plot the results ----------------------------------- ##

using GLMakie
GLMakie.activate!()

ts = range(t0,tf,length=1001);

ρ1 = x2rho.(ξ.x(t)[1:8] for t in ts)
ρ2 = x2rho.(ξ.x(t)[9:16] for t in ts)

p11 = zeros(length(ts))
p12 = zeros(length(ts))
p13 = zeros(length(ts))
p21 = zeros(length(ts))
p22 = zeros(length(ts))
p23 = zeros(length(ts))

for i in 1:length(ts)
    p11[i] = real([1 0 0]*ρ1[i]*[1;0;0])[1]
    p12[i] = real([0 1 0]*ρ1[i]*[0;1;0])[1]
    p13[i] = real([0 0 1]*ρ1[i]*[0;0;1])[1]
    p21[i] = real([1/sqrt(2) 0 -1im/sqrt(2)]*ρ2[i]*[1/sqrt(2);0;1im/sqrt(2)])[1]
    p22[i] = real([0 1 0]*ρ2[i]*[0;1;0])[1]
    p23[i] = real([1/sqrt(2) 0 1im/sqrt(2)]*ρ2[i]*[1/sqrt(2);0;-1im/sqrt(2)])[1]
end

fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, ts, real(p11); color=:red, linewidth=2, label = "|0⟩")
lines!(ax1, ts, real(p12); color=:blue, linewidth=2, label = "|1⟩")
lines!(ax1, ts, real(p13); color=:green, linewidth=2, label = "|2⟩")
axislegend(ax1, position = :lc)

ax2 = Axis(fig[2, 1])
lines!(ax2, ts, real(p21); color=:red, linewidth=2, label = "|R⟩")
lines!(ax2, ts, real(p22); color=:blue, linewidth=2, label = "|1⟩")
lines!(ax2, ts, real(p23); color=:green, linewidth=2, label = "|L⟩")
axislegend(ax2, position = :rc)

display(fig)