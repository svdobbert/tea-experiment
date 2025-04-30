
function transform_data(
    df::DataFrame,
    sampling_dates::DataFrame,
    env_var::String,
    date_col::String,
    exposure_time::Number,
    retrieving_year::Number,
    group_by::String,
    date_format_in::String
)
    """
    Prepares the DataFrame, trunctuating and transforming the data

    Parameters:

    Returns:
    - DataFrame: The processed DataFrame.
    """
    df = filter(row -> row[Symbol("env_var")] == env_var, df)
    df = select(df, Not(Symbol("env_var")))
    df_trunctuated = prepare_env_data(df, sampling_dates, date_col, exposure_time, retrieving_year, date_format_in)

    if group_by == "month"
        df_trunctuated.group = Dates.month.(df_trunctuated[!, Symbol(date_col)])
        df_grouped = combine(groupby(df_trunctuated, :group), names(df_trunctuated, Not(Symbol(date_col), :group)) .=> mean)
        group = unique(string.(sort(df_trunctuated.group)))

        month_map = Dict(
            "1" => "Jan", "2" => "Feb", "3" => "Mar",
            "4" => "Apr", "5" => "May", "6" => "Jun",
            "7" => "Jul", "8" => "Aug", "9" => "Sep",
            "10" => "Oc", "11" => "Nov", "12" => "Dec"
        )
        group = [month_map[m] for m in group]
    end

    if group_by == "year"
        df_trunctuated.group = Dates.year.(df_trunctuated[!, Symbol(date_col)])
        df_grouped = combine(groupby(df_trunctuated, :group), names(df_trunctuated, Not(Symbol(date_col), :group)) .=> mean)
        group = unique(string.(df_trunctuated.group))
    end

    if group_by == "all"
        df_trunctuated.group = fill("all", nrow(df_trunctuated))
        df_grouped = combine(groupby(df_trunctuated, :group), names(df_trunctuated, Not(Symbol(date_col), :group)) .=> mean)
        group = unique(string.(df_trunctuated.group))
    end
    df_transposed = DataFrame(permutedims(df_grouped))[2:end, :]

    rename!(df_transposed, group)

    return df_transposed
end

function random_forest_importance(
    df_env::DataFrame,
    df_tea::DataFrame,
    sampling_dates::DataFrame,
    data_id::String,
    date_col::String,
    id_col::String,
    exposure_time::Number,
    retrieving_year::Number,
    season::String="all",
    plot=true,
    save_pdf::Bool=false,
    save_png::Bool=false,
    group_by::String="all",
    date_format_in::String="dd.mm.yyyy HH:MM"
)
    """
    Trunctuates a Dataframe with a datetime column to a specific sub-dataframe.

    Parameters:

    Returns:
    - Feature importance as .csv and (optional) plot.
    """

    parts = split(data_id, "_")
    if exposure_time == 0
        exposure_time = parse(Int, replace(parts[2], "M" => ""))
        @info "Exposure time was set to $exposure_time"
    end

    if retrieving_year == 0
        retrieving_year = parse(Int, "20" * parts[3])
        @info "Retrieving year was set to $retrieving_year"
    end

    df_at_transformed = transform_data(df_env, sampling_dates, "AT", date_col, exposure_time, retrieving_year, group_by, date_format_in)
    rename!(df_at_transformed, Dict(names(df_at_transformed)[i] => "AT_" * string(names(df_at_transformed)[i]) for i in 1:length(names(df_at_transformed))))
    df_st_transformed = transform_data(df_env, sampling_dates, "ST", date_col, exposure_time, retrieving_year, group_by, date_format_in)
    rename!(df_st_transformed, Dict(names(df_st_transformed)[i] => "ST_" * string(names(df_st_transformed)[i]) for i in 1:length(names(df_st_transformed))))
    df_sm_transformed = transform_data(df_env, sampling_dates, "SM", date_col, exposure_time, retrieving_year, group_by, date_format_in)
    rename!(df_sm_transformed, Dict(names(df_sm_transformed)[i] => "SM_" * string(names(df_sm_transformed)[i]) for i in 1:length(names(df_sm_transformed))))

    df_transformed = hcat(df_at_transformed, df_st_transformed, df_sm_transformed)
    df_transformed.position = names(select(df_env, Not(Symbol(date_col), Symbol("env_var"))))

    df_tea_transposed = DataFrame(permutedims(Matrix(select(df_tea, Not(Symbol(id_col))))),
        :auto)
    rename!(df_tea_transposed, df_tea[!, 1])

    df_tea_transposed.id = string.(names(df_tea))[2:end]

    df_Y = normalize_tea_data(df_tea_transposed, "id")

    selected_row = df_Y[df_Y[!, Symbol("id")].==data_id, :]

    if nrow(selected_row) > 1
        throw(ErrorException("There is more than one row with the given ID: $data_id"))
    end

    if nrow(selected_row) < 1
        error("Incorrect ID. The selected ID ($data_id) is not contained in the id column.")
    end

    if !(occursin(string(exposure_time), data_id))
        error("The selected exposure time ($exposure_time) does not match the tea data id $(data_id).")
    end

    if !(occursin(string(retrieving_year)[end-1:end], data_id))
        error("The selected year ($retrieving_year) does not match the tea data id $(data_id).")
    end

    row_vector = collect(selected_row[1, :])

    cleaned_vector = filter(x -> x isa Number, row_vector)

    if nrow(df_transformed) != length(cleaned_vector)
        throw(ErrorException("There are not the correct number of numeric values in the tea dataframe"))
    end

    df_tea_selected = DataFrame()
    df_tea_selected.values = cleaned_vector
    df_tea_selected.position = df_tea[!, Symbol(id_col)]

    
    df_rf_input = innerjoin(df_transformed, df_tea_selected, on=:position)
    
    n = 10  

    too_many_missing = map(col -> count(ismissing, col) > n, eachcol(df_rf_input))
    df_rf_input = df_rf_input[:, Not(too_many_missing)]
    display(df_rf_input)
    df_rf_input = dropmissing(select(df_rf_input, Not(:position)))


    X = df_rf_input[:, Not(:values)]
    x_values = names(X)
    X = Matrix{Float64}(X)
    y = convert(Vector{Float64}, df_rf_input.values)

    #  normalizing X and y 
    std_X = std(X, dims=1)
    std_X[std_X.==0] .= 1
    X = (X .- mean(X, dims=1)) ./ std_X

    std_y = std(y, dims=1)
    std_y = std_y == 0 ? 1 : std_y
    y = (y .- mean(y, dims=1)) ./ std_y

    # remove near constant columns
    X = X[:, std.(eachcol(X)).>1e-6]

    @show any(ismissing, X) || any(ismissing, y)

    # train random forest classifier
    model = RandomForestRegressor(
    n_trees=100,
    max_depth=10,
    n_subfeatures=round(Int, sqrt(size(X, 2))),
    min_samples_leaf=1, 
    partial_sampling=1.0  
)
    # fit model and get feature importances
    model = DecisionTree.fit!(model, X, y)
    importances = impurity_importance(model)
    display(importances)
    feature_names = x_values
    df_importance = DataFrame(feature=feature_names, importance=importances)
    sort!(df_importance, :importance, rev=true)

    # if sum(df_importance.importance) != 1
    #     throw(ErrorException("The sum of the importances is not equal to 1. Please check the model."))
    # end

    CSV.write("./$(data_id)_$(season)_feature_importance.csv", df_importance)

    # plot feature importance
    if plot
        if season == "all"
            season_title = ""
        else
            season_title = ", season = $season"
        end

        title = "$data_id$season_title"
        p = Gadfly.plot(
            df_importance,
            x=:feature,
            y=:importance,
            Geom.bar,
            Guide.xlabel("Feature"),
            Guide.ylabel("Importance"),
            Guide.title(title),
            Coord.cartesian(ymin=0, ymax=1),
            Theme(
                default_color=main_col,
                background_color="white",
                highlight_width=0mm)
        )

        if save_pdf
            draw(PDF("./$(data_id)_$(season)_feature_importance.pdf", 14cm, 14cm), p)
        end

        if save_png
            draw(PNG("./$(data_id)_$(season)_feature_importance.png", 14cm, 14cm), p)
        end

        display(p)
        return p
    end
end
