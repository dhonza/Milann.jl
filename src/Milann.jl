module Milann

using Flux
import Flux.Tracker: data, @grad, track

using Statistics

# This implemens a MIL version where the bag instances are stored in a continuous tensor and bags are delimited by ranges. 

export RangeMIL, segmax, segmean, segmax_naive, segmean_naive

struct RangeMIL
    premodel
    aggregation
    postmodel
end
@Flux.treelike RangeMIL

function (m::RangeMIL)(X::AbstractArray, B::AbstractArray)
    (nfeatures, ninsts), nbags = size(X), length(B)
    # X: nfeatures x ninsts, B: nbags
    Ypre = m.premodel(X)    
    # Ypre: npremodeloutput x ninstances
    Yagg = m.aggregation(Ypre, B)
    # Yagg: npremodeloutput x nbags
    m.postmodel(Yagg)
end

"""
segmax(X, segments)

Compute segmented maximum for instances `X` and bags indexed by `segments`.
"""
function segmax(X, segments)
    Y = similar(X, size(X, 1), length(segments))
        
    # naive approach, fast on CPU
    #     for (i, seg) in enumerate(segments)
    #         Y[:, i:i] .= mean(view(X, :, seg), dims=2)
    #     end
        
    # sliced approach much better for GPU
    Y .= 0
    mlen = maximum(length.(segments))
    for i in 1:mlen
        Yindices = [j for (j, e) in enumerate(segments) if length(e) >= i]
        Xindices = map(q -> first(q) + i - 1, filter(e -> length(e) >= i, segments))
        Y[:, Yindices] = max.(view(Y, :, Yindices), view(X, :, Xindices))   
    end
    Y
end

"""
segmax_idx(X, segments)

Compute segmented maximum for instances `X` and bags indexed by `segments`. Same as segmax, but returns also the indices of maxima in input (used for grad computation).
"""
function segmax_idx(X, segments)
    Y = similar(X, size(X, 1), length(segments))
    Y2 = similar(X, size(Y)...) # twice the memmory!
    idxs = similar(X, size(Y)...)    
    Y .= -Inf
    Y2 .= NaN
    idxs .= 0
    mlen = maximum(length.(segments))
    for i in 1:mlen
        Yindices = [j for (j, e) in enumerate(segments) if length(e) >= i]
        Xindices = map(q -> first(q) + i - 1, filter(e -> length(e) >= i, segments))
        Y2[:, Yindices] = max.(view(Y, :, Yindices), view(X, :, Xindices))
        idxs[:, Yindices] += view(Y, :, Yindices)  .!= view(Y2, :, Yindices) # has been max increased since last iteration?
        tmp = Y; Y = Y2; Y2 = tmp # just copy references
    end
    # idxs now holds indices of maxima relative to bags, make them relative to inputs
    offsets = first.(segments) .- 1 # first indices in X minus 1
    Y, Int.(cpu(idxs) .+ reshape(offsets, 1, length(offsets)))
end
    
segmax(X::TrackedArray, segments) = track(segmax, X, segments)
    
function dsegmax(Δ, segments, idxs)
    grads = similar(Δ, size(Δ, 1), last(last(segments)))
    grads .= 0
    for (i, row) in enumerate(eachrow(idxs))
        # @show size(row), minimum(row), size(Δ)
        grads[i, row] .= Δ[i, :]
    end 
    grads, nothing
end
    
@grad function segmax(X, segments)
    Y, idxs = segmax_idx(data(X), segments)
    Y, Δ -> dsegmax(Δ, segments, idxs)
end

"""
segmean(X, segments)

Compute segmented mean for instances `X` and bags indexed by `segments`.
"""
function segmean(X, segments)
    Y = similar(X, size(X, 1), length(segments))
        
    # naive approach, fast on CPU
    #     for (i, seg) in enumerate(segments)
    #         Y[:, i:i] .= mean(view(X, :, seg), dims=2)
    #     end
        
    # sliced approach much better for GPU
    Y .= 0
    mlen = maximum(length.(segments))
    for i in 1:mlen
        Yindices = [j for (j, e) in enumerate(segments) if length(e) >= i]
        Xindices = map(q -> first(q) + i - 1, filter(e -> length(e) >= i, segments))
        Y[:, Yindices] += view(X, :, Xindices)
    end
    Y ./= reshape(gpu(length.(segments)), 1, length(segments))
    Y
end
    
segmean(X::TrackedArray, segments) = track(segmean, X, segments)
    
function dsegmean(Δ, segments)
    grads = similar(Δ, size(Δ, 1), last(last(segments)))
    
    # naive approach, fast on CPU
    #     for (i, seg) in enumerate(segments)
    #         grads[:, seg] .= view(Δ, :,i) / length(seg)
    #     end
    
    # sliced approach much better for GPU
    mlen = maximum(length.(segments))
    for i in 1:mlen
        Yindices = [j for (j, e) in enumerate(segments) if length(e) >= i]
        Xindices = map(q -> first(q) + i - 1, filter(e -> length(e) >= i, segments))
        grads[:, Xindices] .= view(Δ, :, Yindices)
    end
    for i in 2:mlen # normalization
        Xindices = vcat((collect(seg) for seg in segments if length(seg) == i)...)
        if length(Xindices) > 0
            grads[:, Xindices] ./= i
        end
    end
    
    grads, nothing
end
    
@grad segmean(X, segments) = segmean(data(X), segments), Δ -> dsegmean(Δ, segments)

"""
segmax_naive(X, segments)

Segmented maximum: naive reference implementation using default AD.
"""
function segmax_naive(X, segments)
    hcat((maximum(X[:,s], dims=ndims(X)) for s in segments)...)
end

"""
    segmean_naive(X, segments)

Segmented mean: naive reference implementation using default AD.
"""
function segmean_naive(X, segments)
    hcat((mean(X[:,s], dims=ndims(X)) for s in segments)...)
end

end # module
