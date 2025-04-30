function trunctuate_df(df::DataFrame, sampling_dates::DataFrame, date_col::String, exposure_time::Number, retrieving_year::Number, date_format_in::String="dd.mm.yyyy HH:MM")
    """
    Trunctuates a Dataframe with a datetime column to a specific sub-dataframe.

    Parameters:
    - df::DataFrame: DataFrame to be trunctuated.
    - sampling_dates::DataFrame: DataFrame containing the sampling dates.
    - datecol::String: Name of the column containing the datetime.
    - exposure_time::Number: Exposure time in months.
    - retrieving_year::Number: Year in which the samples were retrieved.
    - date_format_in::String: The format of the input date column. Default is "dd.mm.yyyy HH:MM".

    Returns:
    - DataFrame: The processed DataFrame.  
    """
    date_format = "dd.mm.yyyy HH:MM"
    date_format_in = DateFormat(date_format_in)
    start_year = string(round(Int, retrieving_year - (exposure_time / 12)))

    df[!, Symbol(date_col)] = [Dates.format(
        DateTime(d, date_format_in),
        date_format
    ) for d in df[!, Symbol(date_col)]]
    df[!, Symbol(date_col)] = DateTime.(df[!, Symbol(date_col)], date_format)

    filtered_columns = Dict{Symbol,DataFrame}()

    for row in eachrow(sampling_dates)
        col = Symbol(row.ID)
        end_date = row[Symbol(string(retrieving_year))]
        end_date = DateTime(end_date, date_format)
        start_date = row[Symbol(start_year)]
        start_date = DateTime(start_date, date_format)
        filtered_df = filter(row -> start_date ≤ row[Symbol(date_col)] ≤ end_date, df)

        filtered_columns[col] = DataFrame(
            date=filtered_df[!, Symbol(date_col)],
            values=filtered_df[!, col]
        )
    end

    dfs = collect(values(filtered_columns))

    merged_df = dfs[1]
    
    for df in dfs[2:end]
        merged_df = outerjoin(
            merged_df, df,
            on=:date,
            makeunique=true
            )
    end

    rename!(merged_df, append!([date_col], sampling_dates.ID))

    @info "Including values between $start_year and $retrieving_year."

    return merged_df
end


function is_in_season(date::DateTime, season::String)
    """
    Calculates the meteorolocical season from dates.

    Parameters:
    - date::DateTime: A vector of dates.
    - season::String: The season to check for.

    Returns:
    - Vector: a vector of seasons (as strings)
    """

    month = Dates.month(Date.(date))
    return (season == "winter" && month in (12, 1, 2)) ||
           (season == "spring" && month in (3, 4, 5)) ||
           (season == "summer" && month in (6, 7, 8)) ||
           (season == "autumn" && month in (9, 10, 11)) ||
           (season == "all" && month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
end


function count_frequencies_range(df::DataFrame, date_col::String, range_values)
    """
    Counts frequencies around specific values in a range

    Parameters:
    - df::DataFrame: DataFrame with original values.
    - datecol::String: Name of the column containing the datetime.
    - range_values: Vector containing values around which values are counted.

    Returns:
    - DataFrame: The processed DataFrame, containing values as columns and frequencies for each site as rows.
    """

    df_counts = DataFrame()
    range_pairs = [(range_values[i-1], range_values[i+1]) for i in 2:length(range_values)-1]

    for col in names(df)
        if col != date_col
            counts = [count(x -> !ismissing(x) && lower ≤ x < upper, df[!, col])
                      for (lower, upper) in range_pairs]
            df_counts = vcat(df_counts, DataFrame(counts', :auto))
        end
    end

    range_names = [Symbol("$(range_values[i])") for i in 2:length(range_values)-1]

    rename!(df_counts, range_names)

    df_counts.position = names(select(df, Not(Symbol(date_col))))

    return df_counts
end


function count_frequencies(df::DataFrame, date_col::String, range_values)
    """
    Counts frequencies above/below specific values in a range

    Parameters:
    - df::DataFrame: DataFrame with original values.
    - datecol::String: Name of the column containing the datetime.
    - range_values: Vector containing values above/below which values are counted.

    Returns:
    - DataFrame: The processed DataFrame, containing values as columns and frequencies for each site as rows.
    """

    df_counts = DataFrame()

    for col in names(df)
        if col != date_col
            counts = [count(x -> !ismissing(x) && (value ≥ 0 ? x ≥ value : x ≤ value), df[!, col])
                      for value in range_values]
            df_counts = vcat(df_counts, DataFrame(counts', :auto))
        end
    end

    range_names = [Symbol("$(range_values[i])") for i in 1:length(range_values)]

    rename!(df_counts, range_names)

    df_counts.position = names(select(df, Not(Symbol(date_col))))

    return df_counts
end

function prepare_env_data(
    df::DataFrame,
    sampling_dates::DataFrame,
    date_col::String,
    exposure_time::Number,
    retrieving_year::Number,
    date_format_in::String="dd.mm.yyyy HH:MM",
)
    """
    Prepares the DataFrame, trunctuating data with different sampling dates seperately to a specific time span before the sampling date.

    Parameters:
    - df::DataFrame: DataFrame to be trunctuated.
    - sampling_dates::DataFrame: DataFrame containing the sampling dates.
    - datecol::String: Name of the column containing the datetime.
    - exposure_time::Number: Exposure time in months.
    - retrieving_year::Number: Year in which the samples were retrieved.
    - date_format_in::String: The format of the input date column. Default is "dd.mm.yyyy HH:MM".

    Returns:
    - DataFrame: The processed DataFrame.  
    """
    df_trunctuated = trunctuate_df(df, sampling_dates, date_col, exposure_time, retrieving_year, date_format_in)
    
    df_trunctuated = filter(row -> is_in_season(row[Symbol(date_col)], season), df_trunctuated)


    if nrow(df_trunctuated) < 1
        throw(ErrorException("There are no values within the selected time and season: $season"))
    end

    return df_trunctuated
end

function normalize_tea_data(df::DataFrame, id_col::String)
    """
    Normalizes and cleans the a DataFrame

    Parameters:
    - df::DataFrame: DataFrame containing tea data
    - id_col::String: Name of the column containing the site ids.

    Returns:
    - DataFrame: The processed DataFrame.
    """
    for col in names(df)
        df[!, col] = coalesce.(df[!, col], 0)
    end

    Y = Matrix{Float64}(select(df, Not(Symbol(id_col))))
    std_Y = std(Y, dims=1)
    Y = (Y .- mean(Y, dims=1)) ./ std_Y
    df_Y = DataFrame(Y, :auto)
    df_Y[!, Symbol(id_col)] = df[!, Symbol(id_col)]

    return df_Y
end

function prepare_data(
    df_env::DataFrame,
    df_tea::DataFrame,
    sampling_dates::DataFrame,
    date_col::String,
    id_col::String,
    data_id::String,
    exposure_time::Number,
    retrieving_year::Number,
    env_var::String,
    date_format_in::String="dd.mm.yyyy HH:MM",
    season::String="all",
    step::Number=0.1,
    countRange::Bool=true,
    saveFrequencies::Bool=true)
    """
    Trunctuates a Dataframe with a datetime column to a specific sub-dataframe and counts frequencies for a following plsr analysis.

    Parameters:
    - df_env::DataFrame: DataFrame containing the environmental data.
    - df_tea::DataFrame: DataFrame containing the tea data.
    - sampling_dates::DataFrame: DataFrame containing the sampling dates.
    - date_col::String: Name of the column containing the datetime.
    - id_col::String: Name of the column containing the site ids.
    - data_id::String: The ID of the tea data.
    - exposure_time::Number: Exposure time in months.
    - retrieving_year::Number: Year in which the samples were retrieved.
    - env_var::String: The environmental variable to be selected.
    - date_format_in::String: The format of the input date column. Default is "dd.mm.yyyy HH:MM".
    - season::String: The meteorological season which should be included. Can also be "all" to select all seasons.
    - step::Number: The step size for the range values.
    - countRange::Bool: If true, frequencies are counted around specific values in a range.
    - saveFrequencies::Bool: If true, frequencies are saved to a CSV file.

    Returns:
    - DataFrame: The processed DataFrame.
    """
    if !(id_col in names(df_tea))
        error("The selected id column name ($id_col) is not contained in the dataframe.")
    end

    if !any(occursin(env_var, s) for s in df_env[:, Symbol("env_var")])
        error("The selected env_var ($env_var) is not contained in the dataframe. Available values are $(unique(df_env[:, Symbol("env_var")]))")
    end

    df = filter(row -> row[Symbol("env_var")] == env_var, df_env)
    df = select(df, Not(Symbol("env_var")))

    min_val = minimum(map(x -> minimum(skipmissing(x)), eachcol(select(df, Not(Symbol(date_col))))))
    max_val = maximum(map(x -> maximum(skipmissing(x)), eachcol(select(df, Not(Symbol(date_col))))))
    range_values = collect(min_val:step:max_val)

    @info "Environmental data ranging from $min_val to $max_val."

    parts = split(data_id, "_")
    if exposure_time == 0
        exposure_time = parse(Int, replace(parts[2], "M" => ""))
        @info "Exposure time was set to $exposure_time"
    end

    if retrieving_year == 0
        retrieving_year = parse(Int, "20" * parts[3])
        @info "Retrieving year was set to $retrieving_year"
    end

    df_trunctuated = prepare_env_data(df, sampling_dates, date_col, exposure_time, retrieving_year, date_format_in)

    if countRange
        df_frequencies = count_frequencies_range(df_trunctuated, date_col, range_values)
    else
        df_frequencies = count_frequencies(df_trunctuated, date_col, range_values)
    end

    # normalize tea data
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

    if nrow(df_frequencies) != length(cleaned_vector)
        throw(ErrorException("There are no the correct number of numeric values in the tea dataframe"))
    end

    df_tea_selected = DataFrame()
    df_tea_selected.values = cleaned_vector
    df_tea_selected.position = df_tea[!, 1]

    df_plsr_input = innerjoin(df_frequencies, df_tea_selected, on=:position)

    if saveFrequencies
        CSV.write("./$(env_var)_$(data_id)_$(exposure_time)_$(retrieving_year)_$(season)_frequencies.csv", df_plsr_input)
    end

    return df_plsr_input
end
