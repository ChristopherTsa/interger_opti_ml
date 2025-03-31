include("../src/main.jl")
include("../src/main_merge_constrained.jl")
include("../src/main_merge_exact.jl")
include("../src/main_merge_lda.jl")
include("../src/main_merge_simple.jl")
include("../src/main_iterative_algorithm_constrained.jl")
include("../src/main_iterative_algorithm_exact.jl")
include("../src/main_iterative_algorithm_lda.jl")
include("../src/main_iterative_algorithm_simple.jl")


main()

main_merge_simple()
main_merge_constrained()
main_merge_lda()

main_iterative_simple()
main_iterative_constrained()
main_iterative_lda()
main_iterative_exact()