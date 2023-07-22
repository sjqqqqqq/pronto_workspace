using PRONTO
using StaticArrays
using LinearAlgebra

## --------------------- helper functions --------------------- ##

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function inprod(x)
    a = x[1:2]
    b = x[3:4]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a'   a*a'+b*b']
    return P
end

## --------------------- modeling --------------------- ##
@kwdef struct TwoSpin <: Model{4,1}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


@define_f TwoSpin begin
    H0 = [1 0;0 -1]
    H1 = [0 1;1 0]
    mprod(-im*(H0+u[1]*H1))*x
end

@define_l TwoSpin 0.01/2*u'*u

@define_m TwoSpin begin
    Pl = [1 0 0 0;0 0 0 0;0 0 1 0;0 0 0 0]
    1/2*x'*Pl*x
end

@define_Q TwoSpin kq*(I(4) - inprod(x))
@define_R TwoSpin kr*I(1)

PRONTO.Pf(θ::TwoSpin,α,μ,tf) = SMatrix{4,4,Float64}(I(4)-inprod(α))

# must be run after any changes to model definition
resolve_model(TwoSpin)


## --------------------- run optimization --------------------- ##

θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0] # initial state
μ = t->SVector{1}(0.5*sin(t)) # open loop input μ(t)
η = open_loop(θ, x0, μ, τ); # guess trajectory
@time ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## --------------------- outputs --------------------- ##

using MAT

ts = t0:0.001:tf
us = [ξ.u(t)[1] for t∈ts]
file = matopen("Uopt_2spin_10.mat","w")
write(file,"Uopt",us)
close(file)

## --------------------- plots --------------------- ##
using GLMakie

dt = 0.01
T = t0:dt:tf

fig = Figure()

sl = Slider(fig[3,1:2], range=1:18, startvalue=3)
ix = sl.value

ax = Axis(fig[1,1])
ylims!(ax, (-1,1))

for i in 1:4
    x = @lift [data.ξ[$ix].x(t)[i] for t in T]
    lines!(ax, T, x)
end

ax = Axis(fig[1,2])
u = @lift [data.ξ[$ix].u(t)[1] for t in T]
lines!(ax, T, u)


ax = Axis(fig[2,1])
# ylims!(ax, (-1,1))

for i in 1:4
    z = @lift [data.ξ[$ix].x(t)[i] + data.ζ[$ix].x(t)[i] for t in T]
    lines!(ax, T, z)
end

ax = Axis(fig[2,2])
u = @lift [data.ξ[$ix].u(t)[1] for t in T]
lines!(ax, T, u)
v = @lift [data.ξ[$ix].u(t)[1]+data.ζ[$ix].u(t)[1] for t in T]
lines!(ax, T, v)


display(fig)



##
record(fig, "animated.mp4", 1:18) do jx
    sl.value[] = jx
end