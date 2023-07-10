import study_and_analysis as sa
df = sa.load_data()

#LOGISTIC REGRESSION
sa.basic_logistic_regression(df, cutoff=80, rand=0)
sa.print_confusion_matrix(df, cutoff=80, rand=0)


sa.logistic_regression_final(df, plot_the_roc=False)
sa.plot_cutoffs_vs_metrics(df)
sa.plot_conf_matrix_Train()
sa.plot_conf_matrix_Test()
print(sa.final_coefs)
sa.plot_final_coeffs()
sa.get_true_positives()
sa.get_true_negatives()
sa.get_false_positives()
sa.get_false_negatives()
sa.sanity_check_test()