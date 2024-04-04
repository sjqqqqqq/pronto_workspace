using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define helper functions ----------------------------------- ##

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end

# get the ith eigenstate
function x_eig(i)
    α = 10 
    v = -α/4
    N = 4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Recomb <: PRONTO.Model{18,1}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end

@define_f Recomb begin
    α = 10 
    v = -α/4
    N = 4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@define_l Recomb begin
    kl/2*u'*I*u 
end

@define_m Recomb begin
    # P = I(18) - inprod(x_eig(2))
    xf = (x_eig(1) + x_eig(2))/sqrt(2)
    P = I(18) - inprod(xf)
    return 1/2*x'*P*x
end

@define_Q Recomb begin
    x_re = x[1:9]
    x_im = x[10:18]
    ψ = x_re + im*x_im
    return kq*mprod(I(9) - ψ*ψ')
end

@define_R Recomb kr*I(1)

# must be run after any changes to model definition
resolve_model(Recomb)

PRONTO.Pf(θ::Recomb,α,μ,tf) = SMatrix{18,18,Float64}(I(18)-inprod(α))
PRONTO.γmax(θ::Recomb, ζ, τ) = PRONTO.sphere(1, ζ, τ)
PRONTO.preview(θ::Recomb, ξ) = [I(9) I(9)]*(ξ.x.^2)

## ----------------------------------- solve the problem ----------------------------------- ##
# eigenstate |3⟩->|1⟩
# eigenstate |3⟩->(|0⟩ + |1⟩)/√2

θ = Recomb(kl=0.01, kr=1, kq=1)
t0,tf = τ = (0,1.5)
x0 = SVector{18}(x_eig(4))
μ = t->SVector{1}(1.0*sin(t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-3,maxiters=50,show_steps=false);

## ----------------------------------- plot the results ----------------------------------- ##

using GLMakie

fig = Figure()
ts = t0:0.001:tf

ax1 = Axis(fig[1,1], xlabel = "time (ω^{-1})", ylabel = "shaking protocol")
ax2 = Axis(fig[2,1], xlabel = "time (ω^{-1})", ylabel = "momentum distribution") 

lines!(ax1, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[11]^2 for t in ts], linewidth = 1, label = "-6ħk")
lines!(ax2, ts, [ξ.x(t)[3]^2+ξ.x(t)[12]^2 for t in ts], linewidth = 1, label = "-4ħk")
lines!(ax2, ts, [ξ.x(t)[4]^2+ξ.x(t)[13]^2 for t in ts], linewidth = 1, label = "-2ħk")
lines!(ax2, ts, [ξ.x(t)[5]^2+ξ.x(t)[14]^2 for t in ts], linewidth = 1, label = "0")
lines!(ax2, ts, [ξ.x(t)[6]^2+ξ.x(t)[15]^2 for t in ts], linewidth = 1, label = "2ħk")
lines!(ax2, ts, [ξ.x(t)[7]^2+ξ.x(t)[16]^2 for t in ts], linewidth = 1, label = "4ħk")
lines!(ax2, ts, [ξ.x(t)[8]^2+ξ.x(t)[17]^2 for t in ts], linewidth = 1, label = "6ħk")
# axislegend(ax2, position = :rc)

display(fig)

## ----------------------------------- output the results ----------------------------------- ##

using MAT

ts = t0:0.001:tf
us = [ξ.u(t)[1] for t in ts]
file = matopen("recomb_3.mat", "w")
write(file, "Uopt", us)
close(file)