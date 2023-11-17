using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

using GellMannMatrices
## helper functions
function qubit_model(n,H0,Hc,L,x,u)
    A = gellmann(n)
    H = H0 + u[1]*Hc
    ρ = 1/2*I(n)
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

@kwdef struct MixState <: PRONTO.Model{3,1}
    kl::Float64 # stage cost gain
end

@define_f MixState begin
    
    X = [0 1;1 0]   # Pauli matrices
    Y = [0 -im;im 0]
    Z = [1 0;0 -1]

    qubit_model(2,Z,X,zeros(2,2),x,u)

    # A = [X,Y,Z]
    
    # H = Z + u[1]*X  # Hamiltonian 
    # # ρ = 1/2*(I(2) + x[1]*X + x[2]*Y + x[3]*Z)   # density matrix
    # ρ = 1/2*(I(2))
    # for i in 1:3
    #     ρ += 1/2*(x[i]*A[i])
    # end
    # ρ̇ = -im*(H*ρ-ρ*H)   # master equation

    # [
    #     real(tr(ρ̇*X)),
    #     real(tr(ρ̇*Y)),
    #     real(tr(ρ̇*Z)),
    # ]  
 
end

@define_l MixState begin
    kl/2*u'*I*u 
end

@define_m MixState begin
    xf = [0,0,-1]
    1/2*(x-xf)'*(x-xf)
end

@define_Q MixState I(3)

@define_R MixState I(1)

resolve_model(MixState)

PRONTO.Pf(θ::MixState,α,μ,tf) = SMatrix{3,3,Float64}(I(3))

## ----------------------------------- solve the problem ----------------------------------- ##

θ = MixState(kl=0.01)
t0,tf = τ = (0,10)
x0 = SVector{3}([0,0,1])
μ = t->SVector{1}(0.2*sin(t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-5,maxiters=50);

## ----------------------------------- plot the results ----------------------------------- ##

import Pkg
Pkg.activate()
using GLMakie
GLMakie.activate!()

function traj_on_sphere()

    ts = range(t0,tf,length=1001);
    
    x1 = [ξ.x(t)[1] for t∈ts];
    y1 = [ξ.x(t)[2] for t∈ts];
    z1 = [ξ.x(t)[3] for t∈ts];

    n = 100
    u = range(-π,stop=π,length=n);
    v = range(0,stop=π,length=n);

    xs = cos.(u) * sin.(v)';
    ys = sin.(u) * sin.(v)';
    zs = ones(n) * cos.(v)';

    aspect=(1, 1, 1)

    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect) 
    lines!(ax, x1, y1, z1; color=:red, linewidth=2)
    surface!(ax, xs,ys,zs, rstride=4, cstride=4, transparency=true, overdraw=false)

    display(fig)

end

traj_on_sphere()