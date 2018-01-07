from pandas import Series, DataFrame
import pandas as pd
import collections
import uncertainties

class Results(object):
    def __init__(self):
        self.data = DataFrame(columns=('Wert', 'Einheit'))

    def app(self, name, value):
        if (type(value.magnitude) == uncertainties.core.Variable):
            df = DataFrame(collections.OrderedDict({'Wert': pd.Series('{:.1u}'.format(value.magnitude), index=[name]),
             'Einheit': pd.Series((value.units), index= [name])}))
        else:
            df = DataFrame(collections.OrderedDict({'Wert': pd.Series('{:.2f}'.format(value.magnitude), index=[name]),
            'Einheit': pd.Series((value.units), index= [name])}))
        self.data = self.data.append(df)

    def makeresults(self):
        print(self.data)
        self.data.to_csv(path_or_buf='ergebnisse.csv', header=True, index_label = False)
