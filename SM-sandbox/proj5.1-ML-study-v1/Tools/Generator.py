import Tools.PDFs as PDFs

import keras
from keras.layers     import Dense, Dropout, Input
from keras.models     import Model, Sequential
from keras.datasets   import mnist
from tqdm             import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam



def create_generator_network () :
	model = Sequential()
	model.add(Dense(32, input_shape=(1,), activation="relu"))
	model.add(Dense(32, activation="relu"))
	model.add(Dense(32, activation="relu"))
	return model



class Generator :
	def __init__ (self) :
		self.generator = create_generator_network()
	def generate_masses (self, num=1) :
		inputs = 
	def generate_models (self, num=1) :
		masses = self.generate_masses(num)
		return [PDFs.PDF(mass=m) for m in masses]