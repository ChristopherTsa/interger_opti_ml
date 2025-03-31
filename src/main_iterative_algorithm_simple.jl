include("building_tree.jl")
include("utilities.jl")
include("merge.jl")
include("main_merge_simple.jl")
include("shift.jl")
using CSV
using DataFrames
using Dates

function main_iterative_simple()
    # Initialize results DataFrame
    results = DataFrame(
        Dataset=String[], 
        D=Int[], 
        Method=String[], 
        Gamma=Float64[], 
        Clusters=Int[],
        Gap=Float64[],
        Train_Errors=Int[],
        Test_Errors=Int[],
        Time=Float64[],
        Iterations=Int[]
    )

    for dataSetName in ["iris", "seeds", "wine", "breast_cancer", "ecoli"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include(joinpath(@__DIR__, "..", "data", dataSetName * ".txt"))
        
        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train,:]
        Y_train = Y[train]
        X_test = reducedX[test,:]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 120

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            println("\t\t\t- Unsplittable clusters (FU)")
            testMerge_simple(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = false)
            println("\t\t\t- Iterative heuristic (FhS)")
            testIterative_simple(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = false, isExact=false)
            println("\t\t\t- Iterative heuristic (FhS) with shifts")
            testIterative_simple(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = false, isExact=false, shiftSeparations=true)
            println("\t\t\t- Iterative exact (FeS)")
            testIterative_simple(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = false, isExact=true)

#            # Do not apply to the multivariate case in the project             
#            println("\t\tMultivarié")
#            println("\t\t\t- Unsplittable clusters (FU)")
#            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true)
#            println("\t\t\t- Iterative heuristic (FhS)")
#            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true, isExact=false)
#            println("\t\t\t- Iterative exact (FeS)")
#            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true, isExact=true)
        end
    end

    # Create results directory if it doesn't exist
    mkpath("results")

    # Save results to CSV with timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    CSV.write("results/main_iterative_simple_results_$(timestamp).csv", results)
end 

function testIterative_simple(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit::Int=-1, isMultivariate::Bool = false, isExact::Bool=false, shiftSeparations::Bool=false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for gamma in 0:0.2:0.8
        print("\t\t\t", gamma * 100, "%\t\t")
        clusters = simpleMerge(X_train, Y_train, gamma)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap, iterationCount = iteratively_build_tree(clusters, D, X_train, Y_train, classes, multivariate = isMultivariate, time_limit = time_limit, isExact=isExact, shiftSeparations = shiftSeparations)
        
        # Add result to DataFrame
        method = if isExact
            "Exact (FeS)"
        elseif shiftSeparations
            "Heuristic (FhS) with shifts"
        else
            "Heuristic (FhS)"
        end

        push!(results, (
            dataSetName,
            D,
            method,
            gamma,
            length(clusters),
            gap == -1 ? NaN : gap,
            prediction_errors(T,X_train,Y_train, classes),
            prediction_errors(T,X_test,Y_test, classes),
            resolution_time,
            iterationCount
        ))
        
        if gap == -1
            print("???%\t")
        else 
            print(round(gap, digits = 1), "%\t")
        end 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        print(round(resolution_time, digits=1), "s\t")
        println(iterationCount, " iterations")

        if gap == -1
            println("Warning: there is no gap since when the time limit has been reached at the last iteration before CPLEX had found no feasible solution")
        end 
    end
    println() 
end
