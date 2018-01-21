from pandas import Series, DataFrame
import pandas as pd
import collections
import numpy
import uncertainties
import pint
from uncertainties import ufloat
from uncertainties import ufloat_fromstr
from pint import UnitRegistry
import string
ureg = UnitRegistry()
Q_ = ureg.Quantity
def return_int(num):
    num_str = str(num)
    num_str = num_str.split('.')[1]
    num_str = num_str[0:1]
    return int(num_str)


class Latexdocument(object):
    def __init__(self, filename):
        self.name = filename
        self.data = DataFrame(columns=(['tex', 'var']))
    def tabular(self, data, header, places, caption, label):
        with open(self.name, 'w') as f:
            f.write('\\begin{table} \n\\centering \n\\caption{' + caption + '} \n\\label{tab: ' + label + '} \n\\begin{tabular}{')
            for i in range(0, len(data)):
                if (type(data[i][0]) == uncertainties.core.Variable or type(data[i][0]) == uncertainties.core.AffineScalarFunc):
                    f.write('S[table-format=' + str(places[i][0]) + ']@{${}\pm{}$} S[table-format=' + str(places[i][1]) + '] ')
                else:
                    f.write('S ')

            f.write('} \n\\toprule  \n')

            for i in range(0, len(data)):
                if i == len(data) - 1:
                    if (type(data[i][0]) == uncertainties.core.Variable or type(data[i][0]) == uncertainties.core.AffineScalarFunc):
                        f.write('\multicolumn{2}{c}{$' + header[i][0:header[i].find('/')] +  '\:/\: \si{' + header[i][header[i].find('/')+1:] + '}$} \\\ \n')
                    else:
                        f.write('{$' + header[i][0:header[i].find('/')] +  '/ \si{' + header[i][header[i].find('/')+1:] + '}$} \\\ \n')
                else:
                    if (type(data[i][0]) == uncertainties.core.Variable or type(data[i][0]) == uncertainties.core.AffineScalarFunc):
                        f.write('\multicolumn{2}{c}{$' + header[i][0:header[i].find('/')] +  '\:/\: \si{' + header[i][header[i].find('/')+1:] + '}$} & ')
                    else:
                        f.write('{$' + header[i][0:header[i].find('/')] +  '/ \si{' + header[i][header[i].find('/')+1:] + '}$} & ')


            f.write('\\midrule  \n')
            for i in range(0, len(data[0])):
                for j in range(0, len(data)):
                    if (type(data[j][i]) == uncertainties.core.Variable or type(data[j][i]) == uncertainties.core.AffineScalarFunc):
                        if j == len(data) - 1:
                            f.write(('{:.' + str(return_int(places[j][0])) + 'f} ' + '& {:.' + str(return_int(places[j][1])) + 'f}' + '\\\ \n').format(data[j][i].n, data[j][i].s))
                        else:
                            f.write(('{:.' + str(return_int(places[j])) + 'f} ' + '& {:.' + str(return_int(places[j][1])) + 'f}'+ ' & ').format(data[j][i].n, data[j][i].s))
                    else:
                        if j == len(data) - 1:
                            f.write(('{:.' + str(places[j]) + 'f}' + '\\\ \n').format(data[j][i]))
                        else:
                            f.write(('{:.' + str(places[j]) + 'f}' + ' & ').format(data[j][i]))
            f.write('\\bottomrule \n\\end{tabular} \n\\end{table}')

    def app(self, name, value):
            if (type(value.magnitude) == uncertainties.core.Variable or type(value.magnitude) == uncertainties.core.AffineScalarFunc):
                val = '{:+.1uS}'.format(value.magnitude)
                s = '{:Lx}'.format(Q_(2, value.units)) + '~'
                df = DataFrame(collections.OrderedDict({'var': pd.Series(value, index = [name] ),
                #'tex': name + ' = \SI{' + val[:val.index('+')]+ ' \pm ' + val[val.index('-')+1:] + s[s.index('}{'):s.index('~')]}))
                'tex': name + ' = \SI{' + val + '}{' + s[s.index('}{') + 2:s.index('~')]}))
                self.data = self.data.append(df)
            else:
                df = DataFrame({'var': pd.Series(value, index = [name] ),
                'tex': name + ' = ' + '{:Lx}'.format(value)})
                self.data = self.data.append(df)


    def makeresults(self):
        print(self.data['var'])
        with open(self.name, 'w') as f:
            for i in self.data['tex']:
                f.write(i + '\n')
