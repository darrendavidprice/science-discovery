import sys

import numpy as np
import scipy.stats as stats


expected_meas = 57
expected_std  = np.sqrt(expected_meas)
num_toys      = int(1e7)
print("=============================================================")
print("===   Step 1   ==============================================")
print("=============================================================")
print(f"Expected measurement is {expected_meas:.3f}")
print(f"Expected std. dev.   is {expected_std:.3f}")
print("=============================================================")


toys      = np.random.poisson(expected_meas, (num_toys))
toys_mean = np.mean(toys)
toys_std  = np.std (toys)
print("===   Step 2  (throw {} toys)  ".format(num_toys).ljust(61, "="))
print("=============================================================")
print(f"Toys mean      is {toys_mean:.3f}  (precision of {100.*(1.-np.fabs(toys_mean-expected_meas)/expected_meas):.2f}%)")
print(f"Toys std. dev. is {toys_std :.3f}  (precision of {100.*(1.-np.fabs(toys_std-expected_std)/expected_std):.2f}%)")
print("=============================================================")


usual_pulls, ste_pulls = [], []
for idx, toy_meas in enumerate(toys) :
	pull_method_1 = (toy_meas-expected_meas) / np.sqrt(toy_meas-1)
	pull_method_2 = (toy_meas-expected_meas) / expected_std
	usual_pulls.append( pull_method_1 )
	ste_pulls  .append( pull_method_2 )
	if 100*(idx+1) % num_toys != 0 : continue
	sys.stdout.write("\r     processing toys {:.0f}%".format(100.*(idx+1)/num_toys))
	sys.stdout.flush()

print("\r===   Step 3  (compare pulls)  ".ljust(61, "="))
print("=============================================================")
print(f"PULLS METHOD 1:   mean  [std. dev.] is {np.mean(usual_pulls):.3f}  [{np.std(usual_pulls):.3f}]")
print(f"PULLS METHOD 2:   mean  [std. dev.] is {np.mean(ste_pulls):.3f}  [{np.std(ste_pulls):.3f}]")
print("=============================================================")
print("N.B. Method 1 uses sqrt(n) as the estimated std. dev. for a measurement of n events")
print("     Method 2 uses the expected std. dev. as the estimated std. dev. for every toy")
print("     Method 2 is unbiased by definition, since the expected std. dev is what was used to generate the toys")