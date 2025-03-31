include("building_tree.jl")
include("utilities.jl")
include("merge.jl")
using CSV
using DataFrames
using Dates

function main_merge_lda()

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
        Iterations=Int[]  # Keep this column
    )

    for dataSetName in ["iris", "seeds", "wine", "breast_cancer_", "ecoli_"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include("../data/" * dataSetName * ".txt")
        
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
        time_limit = 30

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            testMerge_lda(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = false)
            println("\t\tMultivarié")
            testMerge_lda(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit = time_limit, isMultivariate = true)
        end
    end

    # Create results directory if it doesn't exist
    mkpath("results")
    
    # Save results to CSV with timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    CSV.write("results/main_merge_lda_results_$(timestamp).csv", results)
end 

function testMerge_lda(X_train, Y_train, X_test, Y_test, D, classes, results, dataSetName; time_limit::Int=-1, isMultivariate::Bool = false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for gamma in 0:0.2:1
        print("\t\t\t", gamma * 100, "%\t\t")
        clusters = ldaMerge(X_train, Y_train, gamma)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, classes, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")

        # Add result to DataFrame
        push!(results, (
            dataSetName,
            D,
            isMultivariate ? "Multivariate" : "Univariate",
            gamma,
            length(clusters),
            gap,
            prediction_errors(T,X_train,Y_train, classes),
            prediction_errors(T,X_test,Y_test, classes),
            resolution_time,
            1  # Iterations column with default value 1
        ))
    end
    println() 
end
