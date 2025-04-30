module TeaAnalysis

using Arrow
using CategoricalArrays
using CSVFiles
using Downloads
using DataFrames
using GZip
using Tar
using MultivariateStats
using Plots
using MLJ
using Dates
using Statistics
using Missings
using LinearAlgebra
using Gadfly
using Compose
using ColorSchemes
using Jchemo
using DataFrames
using StatsBase
using CSV
using Loess
using Cairo
using Dates
using DecisionTree

include("constants.jl")
include("load-data.jl")
include("check-data.jl")
include("preprocessing.jl")
include("plsr.jl")
include("postprocessing.jl")
include("plot.jl")
include("random-forest.jl")

# parameters
env_var = "ST" # The environmental variable to be used for the analysis. Can be "AT", "ST", or "SM".
tea = "roibos" # Tea type: Can be "green" or "roibos"
# data_ids = ["G_12M_17", "G_12M_18", "G_12M_19", "G_24M_18", "G_36M_19", "G_48M_21"] # green
data_ids = ["R_12M_17", "R_12M_18", "R_12M_19", "R_24M_18", "R_36M_19", "R_48M_21"] # roibos

retrieving_year = 0 # Year, during which the samples were retrieved. IMPORTANT: Can be set to 0 to get automatically from data_ids
exposure_time = 0 # Exposure time in months (should be 12, 24, 36, or 48), IMPORTANT: Can be set to 0 to get automatically from data_ids
season = "all" # The meteorological season which should be included. Can also be "all" to select all seasons.

date_col = "date time" # Date column name in the dataframes
id_col = "ID" # The id column name in the tea dataframes
date_format_in = "dd.mm.yyyy HH:MM" # The format of the input date column. Default is "dd.mm.yyyy HH:MM"

step = 0.1 # The step size for the selectivity ratio calculation.
countRange = false # If true, the count range is calculated.

n_folds = 10 # Number of folds for cross-validation.
sig_niveau = 0.05 # The significance level for the selectivity ratio.
smooth = 0.15 # The smoothing parameter for the loess smoothing.

saveFrequencies = true # If true, the frequencies (input for PLSR model) are saved.
plot = true # If true, plots are generated.
save_pdf = false # If true, the plots are saved as pdf.
save_png = false # If true, the plots are saved as png.
plot_type = "all" # Can be "all", "points_raw", "points_smoothed", "points_smoothed_with_sig", "line", or "line_sig".
table_form = "vertical" # The form of the output table (.csv). Can be "vertical" or "horizontal".

random_forest = true # If true, a random forest model is used to calculate feature importance.
group_by = "month" # The aggregation for the environmental variables in the random forest model. Can be "month", "year", or "all".

# load environmental data
df_at = read_csv("./data/AT15_TeaExp__JL.csv")
df_st = read_csv("./data/ST15_TeaExp__JL.csv")
df_sm = read_csv("./data/SM15_TeaExp__JL.csv")

df_at.env_var = fill("AT", nrow(df_at))
df_st.env_var = fill("ST", nrow(df_st))
df_sm.env_var = fill("SM", nrow(df_sm))
df_env = vcat(df_at, df_st, df_sm)

# check environmental data
check_input(df_at, date_col, "01.01.2016 01:00", "01.01.2022 00:00")
check_input(df_st, date_col, "01.01.2016 01:00", "01.01.2022 00:00")
check_input(df_sm, date_col, "01.01.2016 01:00", "01.01.2022 00:00")

# load tea data
df_green = read_csv("./data/GreenTea_JL.csv")
df_roibos = read_csv("./data/RoibosTea_JL.csv")
if tea == "green"
    df_tea = df_green
else
    tea == "roibos"
    df_tea = df_roibos
end

# load sampling dates
sampling_dates = read_csv_space("./data/sampling_dates.csv")

# plsr
results = Dict()
vec_rmse = []
for data_id in data_ids
    results[data_id] = get_selectivity_ratio!(
        df_env,
        df_tea,
        sampling_dates,
        date_col,
        id_col,
        data_id,
        exposure_time,
        retrieving_year,
        step,
        n_folds,
        env_var,
        vec_rmse,
        date_format_in,
        season,
        smooth,
        plot,
        save_pdf,
        save_png,
        countRange,
        saveFrequencies,
        plot_type,
        sig_niveau)

    if random_forest
        random_forest_importance(
            df_env,
            df_tea,
            sampling_dates,
            data_id,
            date_col,
            id_col,
            exposure_time,
            retrieving_year,
            season,
            plot,
            save_pdf,
            save_png,
            group_by,
            date_format_in
        )
    end
end
display(vec_rmse)

# postprocessing
process_results(results, env_var, data_id, season, table_form)

end # module TeaAnalysi
