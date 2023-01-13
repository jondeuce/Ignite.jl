using Ignite
using Documenter

DocMeta.setdocmeta!(Ignite, :DocTestSetup, :(using Ignite); recursive=true)

makedocs(;
    modules=[Ignite],
    authors="Jonathan Doucette <jdoucette@physics.ubc.ca, Christian Kames <ckames@physics.ubc.ca>",
    repo="https://github.com/jondeuce/Ignite.jl/blob/{commit}{path}#{line}",
    sitename="Ignite.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jondeuce.github.io/Ignite.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jondeuce/Ignite.jl",
    devbranch="master",
)
