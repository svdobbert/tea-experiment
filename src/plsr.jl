function remove_constant_columns(df::DataFrame)
    non_constant_columns = [length(unique(skipmissing(df[:, col]))) > 1 for col in names(df)]
    return df[:, non_constant_columns]
end

function get_selectivity_ratio!(
    df_env::DataFrame, 
    df_tea::DataFrame, 
    sampling_dates::DataFrame,
    date_col::String,
    id_col::String,
    data_id::String,
    exposure_time::Number,
    retrieving_year::Number,
    step::Number,
    n_folds::Number,
    env_var::String, 
    vec_rmse::Vector,
    date_format_in::String,
    season::String="all", 
    smooth::Number=0.2, 
    plot=true, 
    save_pdf::Bool=false,
    save_png::Bool=false,
    countRange::Bool=true, 
    saveFrequencies::Bool=true, 
    plot_type::String="all", 
    sig_niveau::Number=0.1)
    """
    Trunctuates a Dataframe with a datetime column to a specific sub-dataframe.

    Parameters:
    - df_env::DataFrame: DataFrame containing environmental data.
    - df_tea::DataFrame: DataFrame containing tea data.
    - sampling_dates::DataFrame: DataFrame containing sampling dates.
    - date_col::String: Name of the date column in the DataFrame.
    - id_col::String: Name of the ID column in the DataFrame.
    - data_id::String: ID of the tea data.
    - exposure_time::Number: Exposure time for the tea data.
    - retrieving_year::Number: Year of data retrieval.
    - step::Number: Step size for the analysis.
    - n_folds::Number: Number of folds for cross-validation.
    - env_var::String: Environmental variable to analyze.
    - vec_rmse::Vector: Vector to store RMSE values.
    - date_format_in::String: Date format for the input data.
    - season::String: Season to analyze (default is "all").
    - smooth::Number: Smoothing parameter for the analysis (default is 0.2).
    - plot::Bool: Whether to plot the results (default is true).
    - save_pdf::Bool: Whether to save the plot as a PDF (default is false).
    - save_png::Bool: Whether to save the plot as a PNG (default is false).
    - countRange::Bool: Whether to calculate the count range (default is true).
    - saveFrequencies::Bool: Whether to save the frequencies (default is true).
    - plot_type::String: Type of plot to generate (default is "all").
    - sig_niveau::Number: Significance level for the analysis (default is 0.1).

    Returns:
    - DataFrame: DataFrame containing selectivity ratios (sel_ratio) with significance (significance), p-value (pval), environmental value (x), and smoothed selectivity ratio for plotting (sel_ratio_smooth).
    """
    df_processed = prepare_data(df_env, df_tea, sampling_dates, date_col, id_col, data_id, exposure_time, retrieving_year, env_var, date_format_in, season, step, countRange, saveFrequencies)

    df_cleaned = dropmissing(select(df_processed, Not(:position)))

    X = remove_constant_columns(df_cleaned[:, Not(:values)])
    env_values = names(X)
    x = parse.(Float64, env_values)
    X = Matrix{Float64}(X)
    y = convert(Vector{Float64}, df_cleaned.values)

    if size(X, 2) == 0
        error("All features were removed due to being constant. Check your data preprocessing.")
    end

    #  normalizing X and y 
    std_X = std(X, dims=1)
    std_X[std_X.==0] .= 1  # Replace zero standard deviations with 1 to prevent division by zero
    X = (X .- mean(X, dims=1)) ./ std_X

    std_y = std(y, dims=1)
    std_y = std_y == 0 ? 1 : std_y  # Prevent division by zero for y
    y = (y .- mean(y, dims=1)) ./ std_y

    # Ensure no NaNs or Infs
    X[isnan.(X).|isinf.(X)] .= 0
    y[isnan.(y).|isinf.(y)] .= 0

    # Cross Validation parameters
    n_samples = size(X, 1)
    n_permutations = 10000  # Number of permutations
    n_features = size(X, 2)
    subset_size = round(Int, 0.8 * n_samples)
    folds = [rand(1:n_samples, subset_size) for _ in 1:n_folds]

    # Step 1: Calculate actual selectivity ratios
    selectivity_ratios_all_folds = []
    coefficient_signs_all_folds = []
    rmse_all_folds = []
    p_values = zeros(n_features)

    n_features_original = length(env_values)

    for fold_idx in 1:n_folds
        test_indices = folds[fold_idx]
        train_indices = setdiff(1:n_samples, test_indices)

        # Split data into train and test sets
        X_train = X[train_indices, :]
        y_train = y[train_indices]

        # Perform feature selection (remove near-zero variance features)
        variances = var(X_train, dims=1)
        keep_features = vec(variances .> 1e-6)
        X_train_selected = X_train[:, keep_features]

        println("Remaining features after selection for fold $fold_idx: ", sum(keep_features))

        global nlv_value = min(3, size(X_train, 1) - 1)
        if nlv_value < 1
            error("Not enough samples for PLS (nlv = $nlv_value).")
        end

        # Fit the PLS model 
        pls_model = Jchemo.plssimp(X_train_selected, y_train, nlv=nlv_value, scal=true)

        pred = Jchemo.predict(pls_model, X_train_selected).pred
        rmse = rmsep(pred, y_train)
        # println("RMSE for fold $fold_idx: ", rmse)

        TT = pls_model.TT
        V = pls_model.V
        residuals = residreg(pred, y_train)

        # Total variance in X
        total_variance_X = sum(var(X_train_selected, dims=1))

        # Explained variance ratio per LV
        explained_variance_ratio = TT / total_variance_X

        # Explained variance per feature
        explained_variance_per_feature = vec(sum(V .^ 2 .* TT', dims=2))

        # Residual variance per feature
        residual_variance = vec(var(X_train_selected, dims=1)) .- explained_variance_per_feature

        # Selectivity ratio
        residual_variance[residual_variance.==0] .= 1 # Avoid division by zero
        selectivity_ratios = explained_variance_per_feature ./ residual_variance

        # Compute regression coefficients B
        B = pls_model.R * pls_model.C'

        # Get the sign of the regression coefficients 
        coefficient_signs = sign.(B[:, 1])
        max_coef = maximum(abs.(B[:, 1]))
        min_coef = minimum(abs.(B[:, 1]))

        # println("Max coefficient: ", max_coef)
        # println("Min coefficient: ", min_coef)

        # Create a vector for all features (matching original feature set)
        selectivity_ratios_full = fill(NaN, n_features_original)
        coefficient_signs_full = fill(NaN, n_features_original)

        #         # optional check of dimensions
        #         println("Shape of V: ", size(V))  
        #         println("Shape of TT: ", size(TT))  
        # println("Shape of explained_variance_per_feature: ", size(explained_variance_per_feature))
        # println("Shape of residual_variance: ", size(residual_variance))
        #         println("Shape of selectivity_ratios: ", size(selectivity_ratios))
        # println("Shape of keep_features: ", size(keep_features))
        # println("Number of selected features: ", sum(keep_features))

        # Assign selectivity ratios to the selected features
        selectivity_ratios_full[keep_features] .= vec(selectivity_ratios)
        coefficient_signs_full[keep_features] .= vec(coefficient_signs)
        # Store the selectivity ratios for this fold
        push!(coefficient_signs_all_folds, coefficient_signs_full)
        push!(selectivity_ratios_all_folds, selectivity_ratios_full)
        push!(rmse_all_folds, rmse)
    end

    # Concatenate the selectivity ratios across folds 
    selectivity_ratios_matrix = hcat(selectivity_ratios_all_folds...)
    println("Number of NaN values in the selectivity ratios matrix: ", count(isnan, selectivity_ratios_matrix))
    selectivity_ratios_df = DataFrame(selectivity_ratios_matrix, :auto)
    selectivity_ratios_mean = [mean(skipmissing(row)) for row in eachrow(selectivity_ratios_df)]
    selectivity_ratios_median = [median(skipmissing(row)) for row in eachrow(selectivity_ratios_df)]

    coefficient_signs_matrix = hcat(coefficient_signs_all_folds...)
    coefficient_signs_mean = sign.(mean(coefficient_signs_matrix, dims=2))
    model = loess(x, vec(coefficient_signs_mean), span=0.1)
    coefficient_signs_mean_smooth = Loess.predict(model, x)

    rmse_matrix = hcat(rmse_all_folds...)
    rmse_df = DataFrame(rmse_matrix, :auto)
    rmse_mean = [mean(skipmissing(row)) for row in eachrow(rmse_df)]
    rmse_value = rmse_mean[1]
    push!(vec_rmse, rmse_value)
    @info "RMSE over all folds: $rmse_value"
    
    # Step 2: Permutation test for p-values
    permuted_ratios_all = zeros(n_features, n_permutations)

    for perm_idx in 1:n_permutations
        permuted_y = shuffle(y)

        # Fit PLS model on permuted data
        pls_model_perm = Jchemo.plssimp(X, permuted_y, nlv=nlv_value, scal=true)

        TT_perm = pls_model_perm.TT
        V_perm = pls_model_perm.V

        # Total variance in X 
        total_variance_X = sum(var(X, dims=1))

        # Explained variance ratio per LV (latent variable)
        explained_variance_ratio = TT_perm ./ total_variance_X

        # Explained variance per feature (sum across LV components)
        explained_variance_per_feature = sum(V_perm .^ 2 .* TT_perm', dims=2)

        # Feature variance (based on original X)
        feature_variance = var(X, dims=1)

        # Residual variance per feature
        residual_variance = vec(feature_variance) .- explained_variance_per_feature

        # Avoid division by zero in residual variance
        residual_variance[residual_variance.==0] .= 1

        # Selectivity ratio
        permuted_selectivity_ratios = explained_variance_per_feature ./ residual_variance

        # Store the permuted selectivity ratios for this permutation
        permuted_ratios_all[:, perm_idx] .= vec(permuted_selectivity_ratios)
    end

    # Compute p-values
    p_values = [mean(abs.(permuted_ratios_all[i, :]) .>= abs(selectivity_ratios_mean[i])) for i in 1:n_features]

    selectivity_ratios_with_sign = vec(coefficient_signs_mean_smooth .* selectivity_ratios_median)

    model = loess(x, selectivity_ratios_with_sign, span=smooth)
    selectivity_ratios_smooth = Loess.predict(model, x)

  
    plsr_result = DataFrame(
        sel_ratio=vec(selectivity_ratios_with_sign),
        p_val=p_values,
        significance=p_values .< sig_niveau,
        x=x,
        sel_ratio_smooth=selectivity_ratios_smooth
    )

    plsr_result.explained_var = plsr_result.sel_ratio ./ (abs.(plsr_result.sel_ratio) .+ 1)

    model = loess(x, plsr_result.explained_var, span=smooth)
    plsr_result.explained_var_smooth = Loess.predict(model, x)

    model_input = plsr_result[plsr_result.significance .== true, :]
    model = loess(model_input.x, model_input.explained_var, span=smooth)
    xmin, xmax = extrema(model_input.x)
    plsr_result.explained_var_smooth_sig = [xi ≥ xmin && xi ≤ xmax ? Loess.predict(model, [xi])[1] : missing for xi in x]

    if plot
        p =  plot_selectivity_ratio(plsr_result, data_id, env_var, season, save_pdf, save_png, plot_type)
        display(p)
    end

    CSV.write("./$(env_var)_$(data_id)_$(exposure_time)_$(retrieving_year)_$(season).csv", plsr_result)

    return plsr_result
end
