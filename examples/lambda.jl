using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## helper functions

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

## Three-level Lambda System model

@kwdef struct LambdaSys <: Model{6,4}
    kl::Float64 = 0.1
    ql::Float64 = 0.0
end

@define_f LambdaSys begin
    H0 = [-0.5 0 0;0 0 0;0 0 -0.5]
    H1 = [0 -0.5 0;-0.5 0 0;0 0 0]
    H2 = [0 -0.5im 0;0.5im 0 0;0 0 0]
    H3 = [0 0 0;0 0 -0.5;0 -0.5 0]
    H4 = [0 0 0;0 0 -0.5im;0 0.5im 0]
    
    mprod(-im*(H0 + u[1]*H1 + u[2]*H2 + u[3]*H3 + u[4]*H4))*x
end

@define_l LambdaSys begin

    r = t->tanh(2*t - 5)
    R = t->[0.01 0 0 0;0 r(t)+1.1 0 0;0 0 0.01 0;0 0 0 -r(t)+1.1]
    Q = mprod(diagm([0,1,0]))

    kl/2*u'*R(t)*u + ql/2*x'*Q*x
end

@define_m LambdaSys begin
    xf = [0,0,1,0,0,0]
    1/2*(x-xf)'*I*(x-xf)
end

@define_Q LambdaSys I(6)
@define_R LambdaSys I(4)
PRONTO.Pf(θ::LambdaSys, αf, μf, tf) = SMatrix{6,6,Float64}(I(6))

resolve_model(LambdaSys)

## Compute the optimal solution

θ = LambdaSys(ql=0.1) # instantiate a new model
τ = t0,tf = 0,5 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] # initial state
xf = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] # final state
μ = t -> 1e-3*ones(SVector{4}) # zero input
# η = open_loop(θ, x0, μ, τ) # guess trajectory
η = smooth(θ, x0, xf, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-6); # optimal trajectory

## Plot the results

using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)

ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")


lines!(ax1, ts, [ξ.u(t)[1] for t in ts], linewidth = 2, label = "u1")
lines!(ax1, ts, [ξ.u(t)[2] for t in ts], linewidth = 2, label = "u2")
lines!(ax1, ts, [ξ.u(t)[3] for t in ts], linewidth = 2, label = "u3")
lines!(ax1, ts, [ξ.u(t)[4] for t in ts], linewidth = 2, label = "u4")
axislegend(ax1)
lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[4]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[5]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax2, ts, [ξ.x(t)[3]^2+ξ.x(t)[6]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax2)


display(fig)
