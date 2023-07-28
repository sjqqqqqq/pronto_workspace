using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef
using QuantumControl

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

NX = 4
NU = 1

@kwdef struct Spin2 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


@define_f Spin2 begin
    H0 = 0.5*[1 0;0 -1]
    H1 = [0 1;1 0]
    return mprod(-im * (H0 + u[1]*H1)) * x
end


@define_l Spin2 begin
    kl/2*(u'*I*u) 
end

@define_m Spin2 begin
    P = [1 0 0 0;
         0 0 0 0;
         0 0 1 0;
         0 0 0 0]
    return 1/2*(x'*P*x)
end

@define_Q Spin2 begin
    x_re = x[1:2]
    x_im = x[3:4]
    ψ = x_re + im*x_im
    return kq*mprod(I(2) - ψ*ψ')
end

@define_R Spin2 kr*I(NU)
PRONTO.Pf(θ::Spin2,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# must be run after any changes to model definition
resolve_model(Spin2)


## ------------------------------- demo: |0⟩ -> |1⟩ in 10 ------------------------------- ##

x0 = SVector{NX}([1 0 0 0])
xf = SVector{NX}([0 1 0 0])
t0,tf = τ = (0,5)
tlist = collect(range(0, 5, length=500));

θ = Spin2()
μ = t -> [0.2 * QuantumControl.Shapes.flattop(t, T=5, t_rise=0.3, func=:blackman)];
φ = open_loop(θ,x0,μ,τ)
@profview ξ,data = pronto(θ,x0,φ,τ; tol=1e-4)

## ----------------------------------- output results as MAT ----------------------------------- ##

using MAT

ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_10_1.0.mat","w")
write(file,"Uopt",us)
close(file)

# terminal cost for each iteration
[PRONTO.m(θ,ξ.x(tf),ξ.u(tf),tf) for ξ in data.ξ]

## ----------------------------------- plot results ----------------------------------- ##
using GLMakie

fig = Figure()
ts = 0:0.001:tf
ax = Axis(fig[1,1]; xlabel="time [s]", ylabel="state")
x1 = [data.ξ[end].x(t)[i] for t∈ts, i∈1:4]
foreach(i->lines!(ax, ts, x1[:,i]), 1:4)

ax = Axis(fig[2,1];xlabel="time [s]", ylabel="population")
p = [I(2) I(2)]*x1'.^2
foreach(i->lines!(ax, ts, p[i,:]), 1:2)

ax = Axis(fig[3,1]; xlabel="time [s]", ylabel="u")
u = [data.ξ[end].u(t)[1] for t∈ts]
lines!(ax, ts, u)

display(fig)

##
fig = Figure()
ax = Axis(fig[1,1]; xlabel="iterations", ylabel="distance", yscale=log10)
x_julia = [0.9515, 0.2952, 0.0182, 6.912e-5]
x_John = [0.9515, 0.3165, 0.0158, 8.82e-5]
x_Marco = [0.9515, 0.3218, 0.016, 1e-4]
x_Krotov = [9.24e-1, 8.83e-1, 8.23e-1, 7.37e-1, 6.26e-1, 4.96e-1, 3.62e-1, 2.44e-1, 1.53e-1, 9.2e-2, 5.35e-2, 3.06e-2, 1.73e-2, 9.78e-3, 5.52e-3, 3.11e-3, 1.76e-3, 9.91e-4]
y_julia = scatterlines!(ax, 1:4, x_julia, color=:red, label="Julia")
y_John = scatterlines!(ax, 1:4, x_John, color=:blue, label="John")
y_Marco = scatterlines!(ax, 1:4, x_Marco, color=:green, label="Marco")
y_Krotov = scatterlines!(ax, 1:18, x_Krotov, color=:black, label="Krotov")
hlines!(ax, [1e-3], color=:black, linestyle=:dash)
Legend(fig[1, 2],
    [y_julia, y_John, y_Marco, y_Krotov],
    ["PRONTO.jl", "MATLAB&C++", "MATLAB", "Krotov.jl"])

display(fig)

save("descent_2lvl.png", fig)