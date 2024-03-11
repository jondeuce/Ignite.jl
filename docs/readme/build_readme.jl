using Pkg
ignite_dir = realpath(joinpath(@__DIR__, "../.."))

Pkg.activate(@__DIR__)
Pkg.develop(; path = ignite_dir)
Pkg.update()

using Literate

readme_src = joinpath(@__DIR__, "README.jl")
readme_dst = ignite_dir
Literate.markdown(readme_src, readme_dst; flavor = Literate.CommonMarkFlavor())
