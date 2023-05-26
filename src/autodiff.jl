# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...) = f(x...) 
Zygote.@adjoint checkpoint(f, x...) = f(x...), ȳ -> Zygote._pullback(f, x...)[2](ȳ)