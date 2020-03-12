import matplotlib.pyplot as plt
percentages =  [0.01, 0.03, 0.05, 0.07, 0.1]
percentages = [p*100 for p in percentages ]
print(percentages)

results_fullgrad_AFOC = [0.0459, 0.0705, 0.0811, 0.0890, 0.1153]
results_inputgrad_AFOC = [0.1093, 0.2090, 0.2443, 0.2789, 0.3292]
results_random_AFOC = [0.2641, 0.3959, 0.5034, 0.5424, 0.5771]

results_fullgrad_KL = [0.0155, 0.0265, 0.0354, 0.0519, 0.0852]
results_inputgrad_KL =  [0.1049, 0.2503, 0.3623, 0.4400, 0.6250]
results_random_KL =  [0.5364, 1.2625, 1.8146, 2.1123, 2.3940]

def plot_results(percentages, results_R, results_IG, results_FG, experiment):
    plt.plot(percentages, results_R, linestyle = "--", marker='o', label="Random", color="brown")
    plt.plot(percentages, results_IG, marker = 'o', label = "InputGrad", color="red" )
    plt.plot(percentages, results_FG, marker = 'o', label = "FullGrad", color= "blue")

    plt.xlabel("% pixels removed")
    plt.xscale('log')
    if experiment == "AFOC":
        plt.ylabel("Absolute Fractional Output Change")
        
    elif experiment == "KL-divergence":
        plt.ylabel("KL-divergence value")

    plt.title(f"Removing % least salient pixels vs {experiment}")
    plt.legend()
    plt.show()

def pixel_perburtation_results():
    plot_results(percentages, results_random_AFOC, results_inputgrad_AFOC, results_fullgrad_AFOC, "AFOC")
    plot_results(percentages, results_random_KL, results_inputgrad_KL, results_fullgrad_KL, "KL-divergence")

