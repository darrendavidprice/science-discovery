import numpy as np

import Tools.Images       as Images
import Tools.Smart_pickle as Smart_pickle


store = {}


def get_rings (num_pixels=51, num_simulations=1000) :
	global store
	Images      .rcParams["num_pixels" ] = num_pixels
	Smart_pickle.rcParams["ignore_load"] = False
	rings = Smart_pickle.load(".rings.pickle", num=num_simulations, **Images.rcParams)
	if rings == None :
	    rings = Images.generate_rings(num_simulations, show=False)
	    Smart_pickle.dump(".rings.pickle", rings, num=num_simulations, **Images.rcParams)
	    print(f"{len(rings)} rings saved to file")
	else :
	    print(f"{len(rings)} rings loaded successfully from file")
	store["rings"  ] = rings
	return rings


def get_showers (num_pixels=51, num_simulations=1000) :
	global store
	Images      .rcParams["num_pixels" ] = num_pixels
	Smart_pickle.rcParams["ignore_load"] = False
	showers = Smart_pickle.load(".showers.pickle", num=num_simulations, **Images.rcParams)
	if showers == None :
	    showers = Images.generate_showers(num_simulations, show=False)
	    Smart_pickle.dump(".showers.pickle", showers, num=num_simulations, **Images.rcParams)
	    print(f"{len(rings)} showers saved to file")
	else :
	    print(f"{len(rings)} showers loaded successfully from file")
	store["showers"] = showers
	return showers


def get_dataset (num_pixels=51, num_simulations=1000, do_rings=True, do_showers=True) :
	rings   = None
	showers = None
	if do_rings   : rings   = get_rings  (num_pixels, num_simulations)
	if do_showers : showers = get_showers(num_pixels, num_simulations)
	return rings, showers


def get_labelled_dataset (num_pixels=51, num_simulations=1000, do_rings=True, do_showers=True) :
	rings, showers = get_dataset(num_pixels, num_simulations, do_rings, do_showers)
	
	TOTAL_DS = [(np.array([0.,1.]), im[3]) for im in rings] + [(np.array([1.,0.]), im[3]) for im in showers]
	np.random.shuffle(TOTAL_DS)

	frac_split = [0.5, 0.75, 1]
	idx_split  = [int(frac_split[i]*len(TOTAL_DS)) for i in range(len(frac_split))]

	TRAIN_DS = TOTAL_DS[            :idx_split[0]]
	VAL_DS   = TOTAL_DS[idx_split[0]:idx_split[1]]
	TEST_DS  = TOTAL_DS[idx_split[1]:idx_split[2]]

	TRAIN_X, TRAIN_Y = np.array([p[1] for p in TRAIN_DS]), np.array([p[0] for p in TRAIN_DS])
	VAL_X  , VAL_Y   = np.array([p[1] for p in VAL_DS  ]), np.array([p[0] for p in VAL_DS  ])
	TEST_X , TEST_Y  = np.array([p[1] for p in TEST_DS ]), np.array([p[0] for p in TEST_DS ])

	store["TOTAL_DS"] = TOTAL_DS
	store["TRAIN_DS"] = TRAIN_DS
	store["VAL_DS"  ] = VAL_DS
	store["TEST_DS" ] = TEST_DS
	store["TRAIN_X" ] = TRAIN_X
	store["TRAIN_Y" ] = TRAIN_Y
	store["VAL_X"   ] = VAL_X
	store["VAL_Y"   ] = VAL_Y
	store["TEST_X"  ] = TEST_X
	store["TEST_Y"  ] = TEST_Y

	return TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y


def get_rings_with_coordinates (num_pixels=51, num_simulations=1000) :
	rings = get_rings  (num_pixels, num_simulations)
	
	TOTAL_DS = [(np.array([im[0], im[1], im[2]/100.]), im[3]) for im in rings]
	np.random.shuffle(TOTAL_DS)

	frac_split = [0.5, 0.75, 1]
	idx_split  = [int(frac_split[i]*len(TOTAL_DS)) for i in range(len(frac_split))]

	TRAIN_DS = TOTAL_DS[            :idx_split[0]]
	VAL_DS   = TOTAL_DS[idx_split[0]:idx_split[1]]
	TEST_DS  = TOTAL_DS[idx_split[1]:idx_split[2]]

	TRAIN_X, TRAIN_Y = np.array([p[1] for p in TRAIN_DS]), np.array([p[0] for p in TRAIN_DS])
	VAL_X  , VAL_Y   = np.array([p[1] for p in VAL_DS  ]), np.array([p[0] for p in VAL_DS  ])
	TEST_X , TEST_Y  = np.array([p[1] for p in TEST_DS ]), np.array([p[0] for p in TEST_DS ])

	store["TOTAL_DS"] = TOTAL_DS
	store["TRAIN_DS"] = TRAIN_DS
	store["VAL_DS"  ] = VAL_DS
	store["TEST_DS" ] = TEST_DS
	store["TRAIN_X" ] = TRAIN_X
	store["TRAIN_Y" ] = TRAIN_Y
	store["VAL_X"   ] = VAL_X
	store["VAL_Y"   ] = VAL_Y
	store["TEST_X"  ] = TEST_X
	store["TEST_Y"  ] = TEST_Y

	return TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y