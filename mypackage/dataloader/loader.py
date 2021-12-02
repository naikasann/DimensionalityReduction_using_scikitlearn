import pandas as pd
import yaml
from pprint import pprint

class loader:
	"""
		class : loader
		method : csv_dataload
	"""
	def csv_dataload(datapath:str, attribute_number:int) -> tuple:
		"""
			Reads csv, converts it to a data frame, and reads and processes
			the data up to the specified line as sample data. (Xdata)
			The number of neighbors in the specified row is used as the teacher data.
			(Ydata. ex: In the case of processing 10 data,
			the 11th line is treated as the teacher data.)

			Args:
				datapath: path to the csv file
				attribute_number: number of attributes(Number of data in Xdata)
			Returns:
				Xdata: Sample data for dataframe
				Ydata: Teacher data
		"""
		# read csv data(terget data)
		load_df = pd.read_csv(datapath, header=None)
		# Extracting data
		X_data = load_df[load_df.columns[load_df.columns != load_df.columns[attribute_number]]].values.tolist()
		Y_data = load_df[load_df.columns[attribute_number]].tolist()
		# for debug
		# pprint(X_data)
		# pprint(Y_data)

		return X_data, Y_data

	def category_dataload(yamlpath:str):
		"""
			Reads yaml file, and converts it to a dictionary.

			Args:
				yamlpath: path to the yaml file
			Returns:
				category_list: list of category
		"""
		with open(yamlpath, 'r') as f:
			data = yaml.safe_load(f)
			return data