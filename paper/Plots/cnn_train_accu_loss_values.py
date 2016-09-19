from laplotter import LossAccPlotter
import numpy as np

loss_train = [1.78287,29.1449,7.78841e-07,13.6862,6.53347,0,20.7715,13.6161,22.2351,0.738925,13.0908,0.777441,0.0472846,5.02116,13.1627,18.9485,0.210569,11.624,2.08745,0.0386455,2.96811,3.9002,12.5514,1.50999e-07,8.83681,0.614701,4.8374,2.94422,5.65934,3.79716,2.42393e-07,3.76043,11.6713,19.8563,2.29324,9.05517,2.73297,0.315229,9.68727,6.09771,7.70933,1.86227,3.1735,0.521857,0.00618037,6.86501,6.27954,14.0736,0.000924774,9.39426,0.769922,0.0245002,1.96274,4.7008,6.90015,8.01472e-05,6.67905,1.29383,1.29645,1.74596,4.9427,6.84285,0.333404,4.63594,5.06641,15.5398,2.11646,5.02996,4.77384,0.979819,4.49098,7.96129,4.60795,1.99647,1.46992,0.838725,0.406617,4.2825,5.30763,13.3822,0.679725,6.00381,1.31804,1.93113e-05,1.47814,7.09132,6.39708,0.0904769,3.79986,2.07594,4.28488e-05,1.09619,3.30311,5.84919,6.51683e-07,5.58052,1.29333,13.7187,1.89587,4.39399,2.13659,0.00336869,3.21854,1.88764,5.14016,0.896579,1.84339,1.04903,0.000184667,5.43554,3.35623,12.1924,0.385686,9.45011,1.85467,0.0246067,0.831138,6.914,6.23844,0.795699,1.95747,0.68692,0.845176,1.0877,3.0746,8.70655,0.00318328,4.42509,0.29063,5.10376,0.989006,2.7661,0.830882,0.000447012,4.70886,5.90386,5.171,0.329779,1.71858,2.65224,0.0874849,5.12659,6.53526,8.14225,0.289866,6.99529,1.06769,0.429264,3.2046,3.89446,3.86922,0.525132,0.974393,0.388117,0.0872195,1.61082,3.31041,4.96678,2.1021e-06,6.17585,0.433333,1.33485,2.73664,1.35853,1.71845,1.46236e-05,2.19909,8.86633,12.8545,0.482935,4.31387,1.73655,0.000135461,4.80087,3.72879,6.01963,0.697551,0.589483,1.58111,1.24803,0.380378,1.94606,2.44027,0.332729,0.58631,1.66398,1.51738,0.381653,1.24591,1.21926,0.089968,0.137374,1.49212,0.414554,1.49662,1.86038,0.586595,0.479933,0.0956203,2.13886,4.08323,0.97873,5.89219,3.0324,2.33401,0.915041,1.16065,0.654674,2.64473,1.67508,2.13872,2.07879,1.037,1.61363,3.01924,0.00194727,0.213232,0.990046,1.19904,0.647825,2.37541,2.722,0.868884,0.498483,1.77989,0.176762,0.312136,1.8974,0.0906598,0.87929,0.000377702,3.5025,2.83095,0.756712,3.14806,1.67057,1.04358,0.0455291,1.58032,0.451919,0.70461,0.853231,1.32766,0.306014,0.0767751,1.2256,3.54072,0.125499,1.10718,1.34228,0.00331049,1.22306,1.02556,1.12424,0.730394,0.000257952,2.43206,0.320216,0.00270821,1.38399,1.13758,0.64275,0.304758,0.425686,1.60879,2.12142,2.41066,1.52521,1.14838,0.618358,1.23101,0.211262,0.401648,4.50295,1.65174,0.887649,0.564754,0.714206,0.780962,1.40717,1.56299,1.00119,0.5446,0.309131,2.22433,0.346811,1.47716,0.125209,1.53101,0.985111,0.162529,1.16698,1.59133,0.423444,0.249217,1.22514,0.15732,1.65533,0.683983,0.313613,0.000591039,0.752796,2.10528,0.359425,0.798952,2.03783,2.00352,0.838242,2.56902,0.731644,0.999082,1.72261,1.6797,0.316591,1.43042,0.481146,1.16165,1.44683,1.09819,0.282121,1.14727,0.982391,1.25662,1.93189,1.45891,0.00163681,0.0745168,1.0052,0.106571,1.25017,0.732631,0.272068,0.521974,0.153365,1.46922,0.216946,0.993699,4.56478,1.23891,0.54856,1.1685,2.60148,0.901606,0.762153,0.673157,0.661107,0.747385,0.583753,1.78587,2.27985,0.537414,0.519801,1.3492,1.27548,0.854889,1.06354,2.40819,0.0574775,0.351584,0.94144,0.277534,0.487703,1.59348,0.273829,0.208106,0.0106441,2.37391,2.91178,2.06921,2.95586,1.41766,1.6503,0.569348,1.00752,0.000979348,0.962174,1.35048,0.720073,0.849725,0.281346,0.397568,2.12944,0.184289,0.210439,0.167841,0.123331,0.523199,1.53868,2.51282,0.220077,0.138104,0.349864,0.346058,0.0392949,1.60673,0.736784,0.365483,0.0697581,2.10601,1.64587,0.284739,2.53066,0.317422,0.39104,0.837833,1.14931,0.00133174,0.364254,1.47253,1.49312,0.970698,0.148329,0.710574,1.1011,0.103337,2.04584,0.840574,0.738854,1.38096,0.886936,2.68339,1.03406,0.561051,1.037,0.180332,0.0567668,0.405977,2.51247,2.46447,0.266659,1.18302,1.21353,1.54037,0.431412,0.585637,0.470308,0.822154,1.09625,1.08513,0.924982,0.730777,0.0290513,2.8241,1.0741,0.0278113,1.62353,1.72484,0.291436,0.0615084,0.0688936,0.109038,0.426259,0.676662,0.0296513,1.46585,0.614088,0.818202,0.623282,0.7245,2.95407,0.801136,0.383492,0.82544,1.04582,1.31102,1.3451,1.89279,0.466176,0.880792,0.567139,1.3591,1.19211,0.356127,0.446713,1.63157,1.74161,0.198557,0.573037,1.82283,0.181068,0.71155,0.183613,0.596359,0.56774,1.02815,0.475456,0.434795,0.303731,1.76786,0.147978,0.567042,3.59143,0.237016,1.07907,0.0650334,0.766743,0.613197,1.66673,0.436797,1.13999]
loss_val = [15.9075,7.81171,7.9644,3.49974,3.15895,4.38955,4.15697,6.17876,3.4366,2.46914,3.41426,3.23049,5.30429,3.19208,2.0779,2.56253,3.2619,5.04907,3.0515,2.65809,2.08003,2.77029,3.90597,2.83559,2.96545,1.90817,2.23167,3.83688,3.1674,3.25176,2.02111,1.89907,3.05517,2.24733,3.50986,1.36536,1.36045,1.30455,1.27106,1.29183,1.22332,1.25968,1.20373,1.21101,1.2103,1.16807,1.19542,1.14616,1.15033,1.14135,1.09643,1.12605,1.08845,1.10505,1.09303,1.07197,1.10399,1.05564,1.09022,1.05491,1.03748,1.06291,1.02575,1.04434,1.01115,1.00698,1.02612,0.975953,1.02462,0.981691,1.01163,0.996219,0.988349,0.991318,0.985924,0.985863,0.986974,0.981544,0.98296,0.979165,0.980247,0.988261,0.988905,0.982036,1.04112,1.05818,0.986896,1.00126,1.02393,0.982019,1.03436,1.07273,0.982022,0.98988,0.994627,0.982016,0.99697,0.982016,1.00086,0.982427]
accu_val = [0.509524,0.519048,0.62381,0.72619,0.769047,0.72619,0.709524,0.669048,0.692857,0.802381,0.778571,0.757143,0.704762,0.695238,0.821428,0.795238,0.740476,0.711905,0.7,0.795238,0.828571,0.769048,0.757143,0.723809,0.780953,0.82619,0.816667,0.761905,0.72381,0.773809,0.807143,0.838095,0.795238,0.802381,0.771429,0.864286,0.861905,0.869048,0.869048,0.866667,0.871429,0.871429,0.871429,0.871429,0.869048,0.871429,0.869048,0.869048,0.871429,0.871429,0.87381,0.871429,0.873809,0.873809,0.871429,0.87381,0.873809,0.871429,0.871429,0.871429,0.87381,0.87381,0.87381,0.87381,0.87381,0.87381,0.876191,0.873809,0.876191,0.876191,0.87381,0.87381,0.878572,0.878571,0.878572,0.878571,0.878571,0.87619,0.876191,0.876191,0.87381,0.87381,0.87381,0.878571,0.871429,0.87381,0.876191,0.87619,0.876191,0.878571,0.87619,0.87381,0.878571,0.876191,0.873809,0.878571,0.873809,0.878571,0.87619,0.878571]

plotter = LossAccPlotter(
			title="Text/No-Text classifier loss and accuracy graph",
			save_to_filepath="./cnn_acc_loss.png",
			show_regressions=False,
			show_averages=True,
			show_loss_plot=True,
			show_acc_plot=True,
			show_plot_window=False,
			x_label="Iteration Count")

# add them all
for iteration in range(20000):
	if iteration%40 == 0:	
		# deactivate redrawing after each update
		plotter.add_values(iteration, loss_train=loss_train[iteration/40], redraw=False)

	if iteration%200 == 0:
		# deactivate redrawing after each update
		plotter.add_values(iteration, loss_val=loss_val[iteration/200], acc_val = accu_val[iteration/200], redraw=False)

# redraw once at the end
plotter.redraw()

plotter.block()

