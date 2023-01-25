import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import os


# During initialization, the class creates a DataFrame object from the json file
# (by default, "deviation.json", you can optionally enter your path to file),
# renames the table columns that have the name of the DataFrame class methods.
class DrawingPlots:

    def __init__(self, filename='deviation.json') -> None:
        self.data_frame = pd.read_json(filename)
        self.data_frame = self.data_frame.rename(columns={'mean': 'floor_vs_ceiling_mean',
                                                          'max': 'floor_vs_ceiling_max',
                                                          'min': 'floor_vs_ceiling_min'})
        try:
            os.mkdir('plots')
        except:
            pass

    # calculates the necessary values (average, maximum, minimum column values) for plotting and returns a dictionary
    def necessary_data_elements(self) -> dict:
        necessary_data_dict = {'total_mean_values':
                                   {column: self.data_frame[column].mean() for column in
                                    self.data_frame.columns[3:6]}}

        necessary_data_dict['floor_ceiling_minmax_values'] = {
            column: (self.data_frame[column].min(), self.data_frame[column].max()) for column in
            self.data_frame.columns[6:]}

        return necessary_data_dict

    # draw plots and returns path to plots images png format
    def draw_plots(self) -> str:
        necessary_data_dict = self.necessary_data_elements()
        plots_names = ('Floor mean column dataset (Red line is mean value of column \"mean\")',
                       'Floor max column dataset (Red line is mean value of column \"max\")',
                       'Floor min column dataset (Red line is mean value of column \"min\")',
                       'Ceiling mean column dataset (Red line is mean value of column \"mean\")',
                       'Ceiling max column dataset (Red line is mean value of column \"max\")',
                       'Ceiling min column dataset (Red line is mean value of column \"min\")')

        for name, column, mean_values, min_max in zip(plots_names, self.data_frame.columns[6:],
                                                      cycle(necessary_data_dict['total_mean_values'].values()),
                                                      necessary_data_dict['floor_ceiling_minmax_values'].values()):
            plt.title(name)
            plt.ylim(min_max[0] - 10, min_max[1] + 10)
            plt.scatter(x=self.data_frame.index, y=self.data_frame[column])
            plt.hlines(y=mean_values, xmin=0, xmax=len(self.data_frame[column]), colors='r')
            plt.savefig(f'plots/{name}.png')
            plt.show()

        return os.path.abspath('plots')


d = DrawingPlots()
print(d.draw_plots())
