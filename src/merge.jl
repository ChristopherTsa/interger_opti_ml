include("struct/distance.jl")
using Statistics
using Distances
using Distributions
using StatsBase
using LinearAlgebra

"""
Essaie de regrouper des données en commençant par celles qui sont les plus proches.
Deux clusters de données peuvent être fusionnés en un cluster C s'il n'existe aucune données x_i pour aucune caractéristique j qui intersecte l'intervalle représenté par les bornes minimale et maximale de C pour j (x_i,j n'appartient pas à [min_{x_k dans C} x_k,j ; max_{k dans C} x_k,j]).

Entrées :
- x : caractéristiques des données d'entraînement
- y : classe des données d'entraînement
- percentage : le nombre de clusters obtenu sera égal à n * percentage
 
Sorties :
- un tableau de Cluster constituant une partition de x
"""
function exactMerge(x, y)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)
    
    # Pour chaque distance
    for distance in distances

        # Si les deux données associées ne sont pas déjà dans le même cluster
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        if cId1 != cId2
            c1 = clusters[cId1]
            c2 = clusters[cId2]

            # Si leurs clusters satisfont les conditions de fusion
            if canMerge(c1, c2, x, y)

                # Les fusionner
                merge!(c1, c2)
                for id in c2.dataIds
                    clusterId[id]= cId1
                end

                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end 
        end 
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

"""
Regroupe des données en commençant par celles qui sont les plus proches jusqu'à ce qu'un certain pourcentage de clusters soit atteint

Entrées :
- x : caractéristiques des données
- y : classe des données
- gamma : le regroupement se termine quand il reste un nombre de clusters < n * gamma ou que plus aucun regroupement n'est possible

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function simpleMerge(x, y, gamma)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)

    remainingClusters = n
    distanceId = 1

    # Pour chaque distance et tant que le nombre de cluster souhaité n'est pas atteint
    while distanceId <= length(distances) && remainingClusters > n * gamma

        distance = distances[distanceId]
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        # Si les deux données associées ne sont pas déjà dans le même cluster
        if cId1 != cId2
            remainingClusters -= 1

            # Fusionner leurs clusters 
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

            # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end 

"""
Test si deux clusters peuvent être fusionnés tout en garantissant l'optimalité

Entrées :
- c1 : premier cluster
- c2 : second cluster
- x  : caractéristiques des données d'entraînement
- y  : classe des données d'entraînement

Sorties :
- vrai si la fusion est possible ; faux sinon.
"""
function canMerge(c1::Cluster, c2::Cluster, x::Matrix{Float64}, y::Vector{Any})

    # Calcul des bornes inférieures si c1 et c2 étaient fusionnés
    mergedLBounds = min.(c1.lBounds, c2.lBounds)
    
    # Calcul des bornes supérieures si c1 et c2 étaient fusionnés
    mergedUBounds = max.(c1.uBounds, c2.uBounds)

    n = size(x, 1)
    id = 1
    canMerge = true

    # Tant que l'ont a pas vérifié que toutes les données n'intersectent la fusion de c1 et c2 sur aucune feature
    while id <= n && canMerge

        data = x[id, :]

        # Si la donnée n'est pas dans c1 ou c2 mais intersecte la fusion de c1 et c2 sur au moins une feature
        if !(id in c1.dataIds) && !(id in c2.dataIds) && isInABound(data, mergedLBounds, mergedUBounds)
            canMerge = false
        end 
        
        id += 1
    end 

    return canMerge
end

"""
Test si une donnée intersecte des bornes pour au moins une caractéristique 

Entrées :
- v : les caractéristique de la donnée
- lowerBounds : bornes inférieures pour chaque caractéristique
- upperBounds : bornes supérieures pour chaque caractéristique

Sorties :
- vrai s'il y a intersection ; faux sinon.
"""
function isInABound(v::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64})
    isInBound = false

    featureId = 1

    # Tant que toutes les features n'ont pas été testées et qu'aucune intersection n'a été trouvée
    while !isInBound && featureId <= length(v)

        # S'il y a intersection
        if v[featureId] >= lowerBounds[featureId] && v[featureId] <= upperBounds[featureId]
            isInBound = true
        end 
        featureId += 1
    end 

    return isInBound
end

"""
Fusionne deux clusters

Entrées :
- c1 : premier cluster
- c2 : second cluster

Sorties :
- aucune, c'est le cluster en premier argument qui contiendra le second
"""
function merge!(c1::Cluster, c2::Cluster)

    append!(c1.dataIds, c2.dataIds)
    c1.x = vcat(c1.x, c2.x)
    c1.lBounds = min.(c1.lBounds, c2.lBounds)
    c1.uBounds = max.(c1.uBounds, c2.uBounds)    
end

"""
Effectue un clustering avec contraintes de classe (Constrained K-means)
en utilisant must-link (ML) et cannot-link (CL) constraints.

Entrées :
- x : caractéristiques des données
- y : classe des données
- gamma : contrôle le nombre de clusters (n_clusters = n * gamma)

Sorties :
- clusters : Vector{Cluster} contenant la partition des données
"""
function constrainedMerge(x::Matrix{Float64}, y::Vector, gamma::Float64)
    n = size(x, 1)
    k = ceil(Int, n * gamma)  # desired number of clusters
    
    # Special case: gamma = 0 or k = 1
    # Create one cluster per class
    if gamma == 0 || k == 1
        unique_classes = unique(y)
        clusters = Vector{Cluster}()
        for class in unique_classes
            indices = findall(==(class), y)
            cluster = Cluster()
            cluster.dataIds = indices
            cluster.x = x
            cluster.class = class
            cluster.lBounds = vec(minimum(x[indices,:], dims=1))
            cluster.uBounds = vec(maximum(x[indices,:], dims=1))
            cluster.barycenter = vec(mean(x[indices,:], dims=1))
            push!(clusters, cluster)
        end
        return clusters
    end
    
    # Handle other edge cases
    if k > n
        clusters = Vector{Cluster}()
        for i in 1:n
            push!(clusters, Cluster(i, x, y))
        end
        return clusters
    end
    
    # Initialize k-means++ with class-aware initialization
    class_centroids = Dict()
    for class in unique(y)
        class_indices = findall(==(class), y)
        class_points = x[class_indices, :]
        n_class_clusters = max(1, round(Int, k * length(class_indices) / n))
        class_centroids[class] = kmeans_plus_plus_init(class_points, n_class_clusters)
    end
    
    # Combine all centroids
    centroids = vcat([class_centroids[class] for class in unique(y)]...)
    k = size(centroids, 1)  # Update k to actual number of centroids
    
    assignments = zeros(Int, n)
    
    # K-means iterations with constraints
    for iter in 1:100
        old_assignments = copy(assignments)
        
        # Assign points to nearest centroid of same class
        for i in 1:n
            min_dist = Inf
            best_cluster = 1
            
            for j in 1:k
                # Only consider clusters that have same class as point
                if isempty(findall(==(j), assignments)) || y[i] == y[findfirst(==(j), assignments)]
                    dist = sum((x[i,:] - centroids[j,:]).^2)
                    if dist < min_dist
                        min_dist = dist
                        best_cluster = j
                    end
                end
            end
            assignments[i] = best_cluster
        end
        
        # Update centroids
        for j in 1:k
            points = findall(==(j), assignments)
            if !isempty(points)
                centroids[j,:] = vec(mean(x[points,:], dims=1))
            end
        end
        
        # Check convergence
        if all(assignments .== old_assignments)
            break
        end
    end
    
    # Create final clusters
    clusters = Vector{Cluster}()
    for j in 1:k
        points = findall(==(j), assignments)
        if !isempty(points)
            cluster = Cluster()
            cluster.dataIds = points
            cluster.x = x
            cluster.class = y[points[1]]
            cluster.lBounds = vec(minimum(x[points,:], dims=1))
            cluster.uBounds = vec(maximum(x[points,:], dims=1))
            cluster.barycenter = vec(mean(x[points,:], dims=1))
            push!(clusters, cluster)
        end
    end
    
    return clusters
end

"""
Initialisation k-means++ des centroids
"""
function kmeans_plus_plus_init(x::Matrix{Float64}, k::Int)
    n, d = size(x)
    
    # Verify data size
    if n == 0 || k > n
        return zeros(k, d)
    elseif k == 1
        # Return mean of all points for single cluster
        return reshape(mean(x, dims=1), 1, d)
    end
    
    centroids = zeros(k, d)
    
    # Choose first centroid randomly
    first = rand(1:n)
    centroids[1,:] = x[first,:]
    
    # Select other centroids
    for i in 2:k
        # Calculate minimum squared distances
        min_dists = [minimum([euclidean(x[j,:], centroids[l,:])^2 
                    for l in 1:i-1]) for j in 1:n]
        
        # Handle potential numerical issues
        max_dist = maximum(min_dists)
        if max_dist == 0
            # If all distances are 0, choose randomly
            next_centroid = rand(1:n)
        else
            # Normalize distances to avoid numerical issues
            probs = min_dists / max_dist
            # Ensure no zeros to avoid division issues
            probs = probs .+ 1e-10
            # Normalize to create proper probability distribution
            probs = probs / sum(probs)
            next_centroid = sample(1:n, Weights(probs))
        end
        
        centroids[i,:] = x[next_centroid,:]
    end
    
    return centroids
end

"""
Effectue un clustering guidé par les classes en utilisant LDA pour la réduction de dimension.

Entrées:
- x : caractéristiques des données (Matrix{Float64})
- y : classe des données (Vector)
- gamma : contrôle le nombre de clusters (n_clusters = n * gamma)

Sorties:
- clusters : Vector{Cluster} contenant la partition des données
"""
##############################
# Helper: myLDA 
# (A simple LDA implementation from scratch that returns a real projection matrix)
##############################
function myLDA(x::Matrix{Float64}, y::Vector)
    classes = unique(y)
    n, d = size(x)
    mean_total = vec(mean(x, dims=1))
    S_W = zeros(d, d)
    S_B = zeros(d, d)
    for c in classes
        idx = findall(==(c), y)
        x_c = x[idx, :]  # points for class c
        mean_c = vec(mean(x_c, dims=1))
        S_W += (x_c .- mean_c')' * (x_c .- mean_c')
        diff = mean_c - mean_total
        S_B += length(idx) * (diff * diff')
    end
    # Regularize S_W to avoid singularity.
    reg = 1e-6 * I(d)
    S_W_reg = S_W + reg
    # Solve the generalized eigenvalue problem.
    eigen_decomp = eigen(S_W_reg \ S_B)
    # Force real values.
    eigenvals = real(eigen_decomp.values)
    eigenvecs = real(eigen_decomp.vectors)
    sorted_idx = sortperm(eigenvals, rev=true)
    num_components = min(length(classes) - 1, d)
    W = eigenvecs[:, sorted_idx[1:num_components]]
    return W  # d×num_components projection matrix.
end

##############################
# LDA-Based Clustering: ldaMerge (Rewritten from scratch)
##############################
function ldaMerge(x::Matrix{Float64}, y::Vector, gamma::Float64)
    n, d = size(x)
    k = ceil(Int, n * gamma)  # desired number of clusters

    # --- Special Cases ---
    if gamma == 0 || k == 1
        unique_classes = unique(y)
        clusters = Vector{Cluster}()
        for c in unique_classes
            idx = findall(==(c), y)
            cluster = Cluster()
            cluster.dataIds = idx
            cluster.x = x
            cluster.class = c
            cluster.lBounds = vec(minimum(x[idx, :], dims=1))
            cluster.uBounds = vec(maximum(x[idx, :], dims=1))
            cluster.barycenter = vec(mean(x[idx, :], dims=1))
            push!(clusters, cluster)
        end
        return clusters
    elseif k > n
        clusters = Vector{Cluster}()
        for i in 1:n
            push!(clusters, Cluster(i, x, y))
        end
        return clusters
    end

    # --- LDA Dimensionality Reduction ---
    W = myLDA(x, y)      # d×m projection matrix, where m = min(num_classes-1, d)
    x_reduced = x * W    # n×m

    n_features = size(x_reduced, 2)

    # --- Centroid Initialization using Class-Aware k-means++ ---
    centroids = zeros(k, n_features)
    assignments = zeros(Int, n)
    current_k = 1
    unique_classes = unique(y)
    for c in unique_classes
        idx = findall(==(c), y)
        x_class = x_reduced[idx, :]  # points in this class
        num_points = size(x_class, 1)
        n_req = max(1, round(Int, k * length(idx) / n))
        n_class_clusters = min(n_req, num_points)
        if n_class_clusters > 0
            class_centroids = kmeans_plus_plus_init(x_class, n_class_clusters)
            block_size = size(class_centroids, 1)
            end_idx = current_k + block_size - 1
            if end_idx > k
                block_size = k - current_k + 1
                end_idx = k
                class_centroids = class_centroids[1:block_size, :]
            end
            centroids[current_k:end_idx, :] = class_centroids
            # Assign each point in this class to the nearest new centroid.
            for i in idx
                dists = [sum((x_reduced[i, :] .- centroids[j, :]).^2)
                         for j in current_k:end_idx]
                _, best_idx = findmin(dists)
                assignments[i] = current_k + best_idx - 1
            end
            current_k = end_idx + 1
            if current_k > k
                break
            end
        end
    end
    # If any centroids remain uninitialized, assign them randomly.
    if current_k <= k
        remaining = setdiff(1:n, assignments)
        for j in current_k:k
            random_idx = isempty(remaining) ? rand(1:n) : rand(remaining)
            centroids[j, :] = x_reduced[random_idx, :]
        end
    end

    # --- K-means Iterations in Reduced Space ---
    for iter in 1:100
        old_assignments = copy(assignments)
        # Update centroids.
        for j in 1:k
            pts = findall(==(j), assignments)
            if !isempty(pts)
                centroids[j, :] = vec(mean(x_reduced[pts, :], dims=1))
            end
        end
        # Reassign each point to its closest centroid (within same original class).
        for i in 1:n
            current_class = y[i]
            valid_centroids = filter(j -> begin
                    pts = findall(==(j), assignments)
                    !isempty(pts) && (y[pts[1]] == current_class)
                end, 1:k)
            if !isempty(valid_centroids)
                dists = [sum((x_reduced[i, :] .- centroids[j, :]).^2) for j in valid_centroids]
                _, best_idx = findmin(dists)
                assignments[i] = valid_centroids[best_idx]
            end
        end
        if all(assignments .== old_assignments)
            break
        end
    end

    # --- Build Final Clusters ---
    clusters = Vector{Cluster}()
    for j in 1:k
        pts = findall(==(j), assignments)
        if !isempty(pts)
            cluster = Cluster()
            cluster.dataIds = pts
            cluster.x = x
            cluster.class = y[pts[1]]
            cluster.lBounds = vec(minimum(x[pts, :], dims=1))
            cluster.uBounds = vec(maximum(x[pts, :], dims=1))
            cluster.barycenter = vec(mean(x[pts, :], dims=1))
            push!(clusters, cluster)
        end
    end

    return clusters
end