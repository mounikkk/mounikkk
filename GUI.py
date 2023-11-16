import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def read_table():
    sg.set_options(auto_size_buttons=True)
    filename = sg.popup_get_file(
        'Dataset to read',
        title='Dataset to read',
        no_window=True,
        file_types=(("CSV Files", "*.csv"), ("Text Files", "*.txt")))
    # --- populate table with file contents --- #
    if filename == '':
        return

    data = []
    header_list = []
    colnames_prompt = sg.popup_yes_no('Does this file have column names already?')
    nan_prompt = sg.popup_yes_no('Drop NaN entries?')

    if filename is not None:
        fn = filename.split('/')[-1]
        try:
            if colnames_prompt == 'Yes':
                df = pd.read_csv(filename, sep=',', engine='python')
                # Uses the first row (which should be column names) as columns names
                header_list = list(df.columns)
                # Drops the first row in the table (otherwise the header names and the first row will be the same)
                data = df[1:].values.tolist()
            else:
                df = pd.read_csv(filename, sep=',', engine='python', header=None)
                # Creates columns names for each column ('column0', 'column1', etc)
                header_list = ['column' + str(x) for x in range(len(df.iloc[0]))]
                df.columns = header_list
                # read everything else into a list of rows
                data = df.values.tolist()
            # NaN drop?
            if nan_prompt == 'Yes':
                df = df.dropna()

            return (df, data, header_list, fn)
        except:
            sg.popup_error('Error reading file')
            return


def show_table(data, header_list, fn):
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25, 25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    window = sg.Window(fn, layout, grab_anywhere=False)
    event, values = window.read()
    window.close()


def main():
    df, data, header_list, fn = read_table()
    show_prompt = sg.popup_yes_no('Show the dataset?')
    if show_prompt == 'Yes':
        show_table(data, header_list, fn)


if __name__ == '__main__':
    main()