include("building_tree.jl")
include("utilities.jl")
using CSV
using DataFrames
using Dates

function main()
    # Initialize results DataFrame
    results = DataFrame(
        Dataset=String[], 
        D=Int[], 
        Method=String[], 
        Gap=Float64[],
        Train_Errors=Int[],
        Test_Errors=Int[],
        Time=Float64[]
    )

    # Pour chaque jeu de données
    for dataSetName in ["iris", "seeds", "wine", "blood", "diabetes", "breast_cancer", "ecoli", "glass"]
        
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
        X_train = reducedX[train, :]
        Y_train = Y[train]
        X_test = reducedX[test, :]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes
        println("Attention : le temps est fixé à 30s pour permettre de faire des tests rapides. N'hésitez pas à l'augmenter lors du calcul des résultats finaux que vous intégrerez à votre rapport.")
        time_limit = 30

        # Pour chaque profondeur considérée
        for D in 2:4

            println("  D = ", D)

            ## 1 - Univarié (séparation sur une seule variable à la fois)
            # Création de l'arbre
            print("    Univarié...  \t")
            T, obj, resolution_time, gap = build_tree(X_train, Y_train, D,  classes, multivariate = false, time_limit = time_limit)

            # Add univariate result to DataFrame
            push!(results, (
                dataSetName,
                D,
                "Univariate",
                gap,
                T !== nothing ? prediction_errors(T,X_train,Y_train, classes) : -1,
                T !== nothing ? prediction_errors(T,X_test,Y_test, classes) : -1,
                resolution_time
            ))

            # Test de la performance de l'arbre
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
                print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
            end
            println()

            ## 2 - Multivarié
            print("    Multivarié...\t")
            T, obj, resolution_time, gap = build_tree(X_train, Y_train, D, classes, multivariate = true, time_limit = time_limit)
            
            # Add multivariate result to DataFrame
            push!(results, (
                dataSetName,
                D,
                "Multivariate",
                gap,
                T !== nothing ? prediction_errors(T,X_train,Y_train, classes) : -1,
                T !== nothing ? prediction_errors(T,X_test,Y_test, classes) : -1,
                resolution_time
            ))
            
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
                print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
            end
            println("\n")
        end
    end

    # Create results directory if it doesn't exist
    mkpath("results")
    
    # Save results to CSV with timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    CSV.write("results/decision_tree_results_$(timestamp).csv", results)
end
