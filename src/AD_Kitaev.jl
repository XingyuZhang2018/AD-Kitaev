module AD_Kitaev

using Zygote
using OMEinsum

include("hamiltonianmodels.jl")
include("variationalipeps.jl")
include("observable.jl")

end
