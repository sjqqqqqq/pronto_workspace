using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
## ----------------------------------- global variables ----------------------------------- ##
X=SMatrix{2,2,Float64}([0 1; 1 0]); # X gate matrix
Y=SMatrix{2,2,ComplexF64}([0 -im; im 0]); # Y gate matrix
Z=SMatrix{2,2,Float64}([1 0; 0 -1]); # Z gate matrix
desiredSize = (3, 3); # used to turn 2x2 matrices into 3x3
num_rows_to_add = desiredSize[1] - size(X, 1); #  used to turn 2x2 matrices into 3x3
num_cols_to_add = desiredSize[2] - size(X, 2); #  used to turn 2x2 matrices into 3x3
H=1/sqrt(2)*SMatrix{2,2,Float64}([1 1; 1 -1]); # H gate matrix
H=[H; zeros(Float64, num_rows_to_add, size(H, 2))]; # # used to turn 2x2 matrices into 3x3
H=[H zeros(Float64, desiredSize[1], num_cols_to_add)]; # 3x3 H gate matrix
S=SMatrix{2,2,ComplexF64}([1 0; 0 im]); # S gate matrix
S=[S; zeros(Float64, num_rows_to_add, size(S, 2))]; # used to turn 2x2 matrices into 3x3
S=[S zeros(Float64, desiredSize[1], num_cols_to_add)]; # 3x3 S gate matrix
T=SMatrix{2,2,ComplexF64}([1 0; 0 exp(im*π/4)]) # T gate matrix
T=[T; zeros(Float64, num_rows_to_add, size(T, 2))]; # used to turn 2x2 matrices into 3x3
T=[T zeros(Float64, desiredSize[1], num_cols_to_add)]; # 3x3 T gate matrix
ρ1 = SMatrix{3,3,Float64}([0 1 0; 1 0 0; 0 0 0]); # ρ1 - ρ8 are the new Pauli matrices for 3 energy levels
ρ2 = SMatrix{3,3,ComplexF64}([0 -im 0; im 0 0; 0 0 0]);
ρ3 = SMatrix{3,3,Float64}([1 0 0; 0 -1 0; 0 0 0]);
ρ4 = SMatrix{3,3,Float64}([0 0 0; 0 0 1; 0 1 0]);
ρ5 = SMatrix{3,3,ComplexF64}([0 0 0; 0 0 -im; 0 im 0]);
ρ6 = SMatrix{3,3,Float64}([0 0 1; 0 0 0; 1 0 0]);
ρ7 = SMatrix{3,3,ComplexF64}([0 0 -im; 0 0 0; im 0 0]);
ρ8 = 1/sqrt(3)*SMatrix{3,3,Float64}([1 0 0; 0 1 0; 0 0 -2]);
e1=@SVector [1.0, 0.0, 0.0]; # IC 1: start at eigenstate 0
e2=1/sqrt(2)*@SVector [1.0, 1.0, 0.0]; # IC 2: start at + 
#e2=@SVector [0.0, 0.0, 1.0]
v10=@SVector [e1'*ρ1*e1, e1'*ρ2*e1, e1'*ρ3*e1, e1'*ρ4*e1, e1'*ρ5*e1, e1'*ρ6*e1, e1'*ρ7*e1, e1'*ρ8*e1]; # transform IC 1 to block vector
v20=@SVector [e2'*ρ1*e2, e2'*ρ2*e2, e2'*ρ3*e2, e2'*ρ4*e2, e2'*ρ5*e2, e2'*ρ6*e2, e2'*ρ7*e2, e2'*ρ8*e2]; # transform IC 2 to block vector
x0=real([v10;v20]); # combine both IC
D=SMatrix{3,3,Float64}([0 0 0;0 0 1; 0 0 0]); # cross product of eigenstate 0 and 2 used for dissipation
function gate2xf(U)
    v1T=[e1'*U'*ρ1*U*e1, e1'*U'*ρ2*U*e1, e1'*U'*ρ3*U*e1, e1'*U'*ρ4*U*e1, e1'*U'*ρ5*U*e1, e1'*U'*ρ6*U*e1, e1'*U'*ρ7*U*e1, e1'*U'*ρ8*U*e1];
    v2T=[e2'*U'*ρ1*U*e2, e2'*U'*ρ2*U*e2, e2'*U'*ρ3*U*e2, e2'*U'*ρ4*U*e2, e2'*U'*ρ5*U*e2, e2'*U'*ρ6*U*e2, e2'*U'*ρ7*U*e2, e2'*U'*ρ8*U*e2];
    xf=real([v1T;v2T])  # final condition
    # xf/norm(xf)
end
n=16; # number of states (N=3, so n=2*(3^2-1)
m=1; # number of inputs
## ----------------------------------- define the model ----------------------------------- ##
@kwdef struct Transmon <: Model{n,m}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
    Rl::Float64 = 1.0
    xf::SVector{n,Float64}
    γ::Float64 
    B::SMatrix{3,3,ComplexF64}
end

@define_f Transmon begin
    H0 = [0 0 0; 0 1 0; 0 0 5];   # H0 Matrix
    ρ0=1/2*I(3);  # ρ0 matrix
    H = [0 0.1 0.3; 0.1 0 0.5; 0.3 0.5 0];    # H matrix for all states
    ρ=1/2*[ρ1;;;ρ2;;;ρ3;;;ρ4;;;ρ5;;;ρ6;;;ρ7;;;ρ8];  # ρ matrix for all states
    L=sqrt(γ)*B # dissipation term
    dρ1=-im*(H0*ρ0 - ρ0*H0) + L*ρ0*L' - 1/2*(L'*L*ρ0+ρ0*L'*L) ;  # M0 matrix
    dρ2=-im*(H0*ρ0 - ρ0*H0) + L*ρ0*L' - 1/2*(L'*L*ρ0+ρ0*L'*L);  # M0 matrix 
    for i=1:Int(n/2)
        dρ1=dρ1+(-im*(H0*ρ[:,:,i] - ρ[:,:,i]*H0) + L*ρ[:,:,i]*L' - 1/2*(L'*L*ρ[:,:,i]+ρ[:,:,i]*L'*L))*x[i]
        dρ2=dρ2+(-im*(H0*ρ[:,:,i] - ρ[:,:,i]*H0) + L*ρ[:,:,i]*L' - 1/2*(L'*L*ρ[:,:,i]+ρ[:,:,i]*L'*L))*x[Int(n/2)+i]  # Calculate dρ matrix for all states
    end
    for j=1:m
        dρ1=dρ1-im*u[j]*(H*ρ0 - ρ0*H)# + L*ρ0*L' - 1/2*(L*L'*ρ0+ρ0*L'*L)
        dρ2=dρ2-im*u[j]*(H*ρ0 - ρ0*H)# + L*ρ0*L' - 1/2*(L*L'*ρ0+ρ0*L'*L)  # Calculate dρ matrix for all inputs
    end
    for i=1:Int(n/2)
        for j=1:m
            dρ1=dρ1+(-im*u[j]*(H*ρ[:,:,i] - ρ[:,:,i]*H))*x[i]
            dρ2=dρ2+(-im*u[j]*(H*ρ[:,:,i] - ρ[:,:,i]*H))*x[Int(n/2)+i] # Combine dρ matrices
        end
    end
    
    dx=[]
    for i=1:Int(n/2)
        dx = [dx; 2*real(tr(dρ1*ρ[:,:,i]))]
    end
    for i=1:Int(n/2)
        dx = [dx; 2*real(tr(dρ2*ρ[:,:,i]))]
    end
    
    return 2*π*dx
    
end
@define_l Transmon begin
    1/2*u'*Rl*u
end

@define_m Transmon begin
    1/2*(x-xf)'*(x-xf)
end

@define_Q Transmon kq*I(n)
@define_R Transmon kr*I(m)
resolve_model(Transmon)

# show the population function on each iteration
PRONTO.preview(θ::Transmon, ξ) = ξ.u
PRONTO.γmax(θ::Transmon, ζ, τ) = PRONTO.sphere(sqrt(2), ζ, τ)
PRONTO.Pf(θ::Transmon, αf, μf, tf) = SMatrix{n,n,Float64}(I(n))
## ----------------------------------- solve the problem ----------------------------------- ##
θ = Transmon(Rl=0.01,xf=gate2xf(H), B=D, γ=0) # instantiate a new model
τ = t0,tf = 0,10 # define time domain
μ = t->SVector{1}(0.2*sin(25*t)) # open loop input μ(t)
#μ = t->[0.1] # open loop input μ(t)
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## ----------------------------------- calculating performance ----------------------------------- ##

1/2*(ξ.x(tf)-gate2xf(H))'*(ξ.x(tf)-gate2xf(H))

## ----------------------------------- plotting input ----------------------------------- ##

using GLMakie

fig = Figure()
ts=0:0.001:tf

# plot the input function which solves the problem
ax = Axis(fig[1,1];xlabel="Time [s]", ylabel="Input")
x3=[ξ.u(t)[i] for t∈ts, i∈1:1]
foreach(i->lines!(ax, ts, x3[:,i]),1:1)

display(fig)


save("transmon_input.png",fig)

## ----------------------------------- plotting states ----------------------------------- ##
fig2 = Figure()
ts=0:0.001:tf

# plot the eigenstate coordinates
ax2 = Axis(fig2[1,1];xlabel="time [s]", ylabel="State vector sizes")
x1=[ξ.x(t)[i] for t∈ts, i∈1:8]
lines!(ax2, ts, x1[:,1],label="x1")
lines!(ax2, ts, x1[:,2],label="x2")
lines!(ax2, ts, x1[:,3],label="x3")
lines!(ax2, ts, x1[:,4],label="x4")
lines!(ax2, ts, x1[:,5],label="x5")
lines!(ax2, ts, x1[:,6],label="x6")
lines!(ax2, ts, x1[:,7],label="x7")
lines!(ax2, ts, x1[:,8],label="x8")
axislegend(position = :ct)

# plot the x gate coordinates
ax3 = Axis(fig2[2,1];xlabel="time [s]", ylabel="State vector sizes for gate")
x2=[ξ.x(t)[i] for t∈ts, i∈9:16]
lines!(ax3, ts, x2[:,1],label="x9")
lines!(ax3, ts, x2[:,2],label="x10")
lines!(ax3, ts, x2[:,3],label="x11")
lines!(ax3, ts, x2[:,4],label="x12")
lines!(ax3, ts, x2[:,5],label="x13")
lines!(ax3, ts, x2[:,6],label="x14")
lines!(ax3, ts, x2[:,7],label="x15")
lines!(ax3, ts, x2[:,8],label="x16")
axislegend(position = :ct)

display(fig2)
save("Transmon_states.png",fig2)
## ----------------------------------- plotting summed magnitude of states ----------------------------------- ##



fig3 = Figure()
ts=0:0.001:tf
ax4 = Axis(fig3[1,1];xlabel="time [s]", ylabel="Summed state vector sizes")
m1=sum([(ξ.x(t)[i].^2) for t∈ts, i∈1:8],dims=2)
m2=sum([(ξ.x(t)[i].^2) for t∈ts, i∈9:16],dims=2)
lines!(ax4, ts, m1[:,1],label="x1 up to x8")
lines!(ax4, ts, m2[:,1],label="x9 up to x16")
axislegend(position = :cc)

display(fig3)


save("Transmon_states_magnitude.png",fig3)
## ----------------------------------- 3d plotting ----------------------------------- ##
using GLMakie
function traj_on_sphere()

    ts = range(t0,tf,length=1001);
    
    x1 = [ξ.x(t)[1] for t∈ts];
    x2 = [ξ.x(t)[9] for t∈ts];
    y1 = [ξ.x(t)[2] for t∈ts];
    y2 = [ξ.x(t)[10] for t∈ts];
    z1 = [ξ.x(t)[3] for t∈ts];
    z2 = [ξ.x(t)[11] for t∈ts];

    n = 100
    u = range(-π,stop=π,length=n);
    v = range(0,stop=π,length=n);

    xs = cos.(u) * sin.(v)';
    ys = sin.(u) * sin.(v)';
    zs = ones(n) * cos.(v)';

    aspect=(1, 1, 1)

    fig2 = Figure(; resolution=(1200, 500))
    ax1 = Axis3(fig2[1, 1]; aspect, title="Eigenstate development") 
    lines!(ax1, x1, y1, z1; color = :blue, linewidth = 3, label="Trajectory")#, colormap = :binary)
    scatter!(ax1,x1[1], y1[1],z1[1],color= :grey, marker= :circle, markersize = 20, label="Initial state") # plot initial point 
    scatter!(ax1,x1[length(ts)],y1[length(ts)],z1[length(ts)],color= :black, marker= :xcross, markersize = 20, label="Final state") # plot final point
    surface!(ax1, xs,ys,zs, rstride=4, cstride=4, transparency=true, overdraw=false)
    axislegend(position = :cb)
    ax2 = Axis3(fig2[1, 2]; aspect, title="Gate development")
    lines!(ax2, x2, y2, z2; color = :red, linewidth = 3, label="Trajectory")#, colormap = :autumn1)
    scatter!(ax2,x2[1], y2[1],z2[1],color= :grey, marker= :circle, markersize = 20,  label="Initial state") # plot initial point 
    scatter!(ax2,x2[length(ts)],y2[length(ts)],z2[length(ts)],color= :black, marker= :xcross, markersize = 20, label="Final state") # plot final point
    surface!(ax2, xs,ys,zs, rstride=4, cstride=4, transparency=true, overdraw=false)
    axislegend(position = :cb)
    display(fig2)
    save("Transmon_sphere.png",fig2)
end

traj_on_sphere()
