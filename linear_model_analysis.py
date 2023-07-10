import study_and_analysis as sa
df = sa.load_data()
# set nice view options for terminal viewing

#LINEAR REGRESSION  
sa.linear_regression_initial(df)
sa.undersample_plot(df)
sa.get_stats(df)
df_samples = sa.random_under_sampler(df, 80)
sa.linear_regression_initial(df_samples)
sa.plot_hist(df_samples)
#search_artist_track_name(df, "Chain", "Some")
df_cols =sa.add_cols(df, 80)
'''
df_split = sa.split_sample_combine(df_cols, cutoff=55, rand=0)
sa.linear_regression_final(df_split, show_plots=True)
df_split = sa.split_sample_combine(df_cols, cutoff=65, rand=0)
sa.linear_regression_final(df_split, show_plots=True)
df_split = sa.split_sample_combine(df_cols, cutoff=75, rand=0)
sa.linear_regression_final(df_split, show_plots=True)
'''
df_split = sa.split_sample_combine(df_cols, cutoff=85, rand=0)
sa.linear_regression_final(df_split, show_plots=True)
###RMSE
sa.linear_regression_sklearn(df_split, show_plots=True)
#df_split = sa.split_sample_combine(df_cols, cutoff=90, rand=0)
#sa.linear_regression_final(df_split, show_plots=True)