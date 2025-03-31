include("../src/main.jl")
include("../src/main_merge_constrained.jl")
#include("../src/main_merge_exact.jl")
include("../src/main_merge_lda.jl")
#include("../src/main_merge_simple.jl")
#include("../src/main_iterative_algorithm_constrained.jl")
#include("../src/main_iterative_algorithm_exact.jl")
#include("../src/main_iterative_algorithm_lda.jl")
#include("../src/main_iterative_algorithm_simple.jl")
#using Base.Threads

#main()

main_merge_constrained()
main_merge_lda()
#main_merge_simple()

# Run the three functions in parallel using tasks
#task1 = @spawn main_iterative_constrained()
#task2 = @spawn main_iterative_lda()
#task3 = @spawn main_iterative_simple()

# Wait for all tasks to complete
#fetch(task1)
#fetch(task2)
#fetch(task3)

main_iterative_exact()

println("All parallel tasks completed")