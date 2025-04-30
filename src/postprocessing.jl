function process_results(results::Dict, env_var::String, data_id::String, season::String, sort::String=vertical)
    """
    Restructures the results of the previous analysis and saves the to a csv file

    Parameters:
    - results::Dict: The results of the previous analysis.
    - env_var::String: environmental variable to process (either "AT", "ST", "SM")
    - data_id::String: The ID of the tea data.
    - season::String: The meteorolocical season which should be included. Can also be "all" to select all seasons.
    - sort::String: The sorting of the data. Can be either "vertical" or "horizontal".

    Returns:
    - DataFrame: The processed DataFrame.
    """

    if sort === "horizontal"
        all_x_values = unique(vcat([df.x for df in values(results)]...))
        sort!(all_x_values)

        df_full = DataFrame(x=all_x_values)

        # Perform outer join iteratively
        df_merged = df_full
        for (data_id, result) in results
            df_result = DataFrame()
            df_result[!, Symbol("$(data_id)_sel_ratio")] = result.sel_ratio
            df_result[!, Symbol("$(data_id)_p_val")] = result.p_val
            df_result[!, Symbol("$(data_id)_significance")] = result.significance
            df_result[!, Symbol("$(data_id)_sel_ratio_smooth")] = result.sel_ratio_smooth
            df_result[!, Symbol("$(data_id)_explained_var")] = result.explained_var
            df_result[!, Symbol("$(data_id)_explained_var_smooth")] = result.explained_var_smooth
            df_result[!, :x] = result.x
            df_merged = outerjoin(df_merged, df_result, on=:x)
        end

        df_merged .= coalesce.(df_merged, NaN)
    end

    if sort === "vertical"
        df_merged = DataFrame()
        for (data_id, result) in results
            df_result = DataFrame(result)
            df_result.tea_ID = fill(data_id, nrow(df_result))
            df_merged = vcat(df_merged, df_result)
        end

        df_merged .= coalesce.(df_merged, NaN)
    end

    formatted_time = Dates.format(now(), "yyyy-mm-dd_HH_MM")
    CSV.write("./$(formatted_time)_$(env_var)_$(season).csv", df_merged)
end


data_id = "G_12M_17"