using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct MixState <: PRONTO.Model{3,1}
    kl::Float64 # stage cost gain
end

@define_f MixState begin
    
    X = [0 1;1 0]   # Pauli matrices
    Y = [0 -im;im 0]
    Z = [1 0;0 -1]
    
    H = Z + u[1]*X  # Hamiltonian 
    ρ = 1/2*(I(2) + x[1]*X + x[2]*Y + x[3]*Z)   # density matrix

    [
        real(tr(-im*(H*ρ-ρ*H)*X)),
        real(tr(-im*(H*ρ-ρ*H)*Y)),
        real(tr(-im*(H*ρ-ρ*H)*Z)),
    ]  
 
end
