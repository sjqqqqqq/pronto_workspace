using PRONTO
using StaticArrays
using LinearAlgebra

@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 # length of pendulum (m)
    g::Float64 = 9.81 # gravity (m/s^2)
end

@define_f InvPend begin
[
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]
end

@define_l InvPend 1/2*x'*I(2)*x + 1/2*u'*I(1)*u


@define_m InvPend begin
    P = [
            1 0;
            0 1;
        ]
    1/2*x'*P*x
end

@define_Q InvPend diagm([10, 1])
@define_R InvPend diagm([1e-3])

resolve_model(InvPend)

##

θ = InvPend(g = 24.79) 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]

α = t->xf
μ = t->u0
η = closed_loop(θ,x0,α,μ,τ)

ξ,data = pronto(θ,x0,η,τ; maxiters=100, tol=1e-4);

##

using GLMakie

fig = Figure()
ts = 0:0.01:tf
ax = Axis(fig[1,1]; xlabel="time [s]", ylabel="angle [rad]",title = "initial curve")
x1 = [0*η.x(t)[1] for t∈ts]
lines!(ax, ts, x1)

ax = Axis(fig[2,1];xlabel="time [s]", ylabel="angular velocity [rad/s]")
x2 = [0*η.x(t)[2] for t∈ts]
lines!(ax, ts, x2)

ax = Axis(fig[3,1]; xlabel="time [s]", ylabel="input [Nm]")
u = [0*η.u(t)[1] for t∈ts]
lines!(ax, ts, u)
display(fig)

save("initial_curve.png", fig, px_per_unit = 2)

##
# using GLMakie

n = Observable(1)
fig = Figure()
ax = Axis(fig[1,1], title = "inverted pendulum on jupiter",ylabel = "x1 [rad]")
T = LinRange(τ...,10000)
x1 = @lift [data.ξ[$n].x(t)[1] for t in T]
lines!(ax, T, x1)
ylims!(ax,-7,7)

x2 = @lift [data.ξ[$n].x(t)[2] for t in T]
ax = Axis(fig[2,1],ylabel = "x2 [rad/s]")
lines!(ax, T, x2)
ylims!(ax,-7,7)

u1 = @lift [data.ξ[$n].u(t)[1] for t in T]
ax = Axis(fig[3,1],xlabel="time [s]", ylabel="u [Nm]")
lines!(ax, T, u1)
ylims!(ax,-15,15)
display(fig)

record(x->n[]=x, fig, "test_jupiter.mp4", 1:length(data.ξ))