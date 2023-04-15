#!/usr/bin/env python
# coding: utf-8

# In[148]:
"""
Python module for reading, processing and plotting Levylab tdms files
"""

"""
Prerequisite Packages:

import os
import shutil
import re
import math
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "notebook"
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
font = {'family' : 'Arial',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams["figure.dpi"] = 600

from scipy import signal
from scipy.optimize import curve_fit
from scipy import polyfit
import random

import json
from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject, TdmsFile
from tqdm import tqdm
from datetime import datetime
import pickle
from sklearn.cluster import KMeans
"""


# In[159]:


class Scan2D():
    def __init__(self, folder = '03-Sweep B IV',\
                 metadata = dict(scantype = 'IV', groupsize = 1, monodirection = 'monotonic',\
                                 convertfilename = False, convertparam = dict(start = 0, step = 1),\
                                 dataofinterest = ['Magnet', 'Temperature', 'AO1', 'AI2', 'AI3', 'AI4'],\
                                 outersweepchannel = 'Magnet', innersweepchannel = 'AO1',\
                                 source = 'AO1', sourceamp = 0.001, Vplus = 'AI4', Vminus = 'AI3', Iminus = 'AI2')):
        """
        initiate a 2D sweep or 1D sweep class

        self.folder = exprimental folder the tdms files are located
        self.metadata['scantype'] = 'IV' or 'Conductance'
        self.metadata['groupsize'] = how many curves taken at the same condition to be averaged
        self.metadata['monodirection'] = 'monoup' 'monodown' 'increasefrom0' 'decreaseto0'
        self.metadata['convertfilename'] = True or False, convert tdms file index to sweep index, in case the sweep index is not saved
        self.metadata['dataofinterest'] = list of channel names you are interested in
        self.metadata['outersweepchannel'] = outer sweep channel (you can choose "datapoint" if your sweep is 1D)
        self.metadata['innersweepchannel'] = inner sweep channel
        self.metadata['source' ... 'Iminus'] = device measurement config,
        note:
        for conductance measurements, use 'X1' ... 'X8' as channel names
        for IV measurements, use 'AI1' ... 'AI8' as channel names
        """

        self.folder = './' + folder
        # set the folder which contains the tdms files of one experiment run
        self.filelist = [self.folder + '/' + f for f in os.listdir(self.folder) if f.endswith('.tdms')]
        # create a list of files which end with '.tdms'
        self.filelist = sorted(self.filelist)
        
        self.metadata = metadata
        self.metadata['pivottablelowpassfiltered'] = False
        self.metadata['pivottablecolumn'] = ''
        self.metadata['pivottablerow'] = ''
        self.metadata['pivottablevalue'] = ''
        self.metadata['dzdx_columnstep'] = 0
        self.metadata['dzdy_rowstep'] = 0
        # set metadata
        
#         self.metadata['scantype'] = scantype
#         self.metadata['groupsize'] = groupsize
#         self.metadata['monodirection'] = monodirection
#         self.metadata['convertfilename'] = convertfilename
#         self.metadata['convertparam'] = convertparam
#         self.metadata['dataofinterest'] = dataofinterest
#         self.metadata['outersweepchannel'] = outersweepchannel
#         self.metadata['innersweepchannel'] = innersweepchannel
#         self.metadata['source'] = source
#         self.metadata['sourceamp'] = sourceamp
#         self.metadata['Vplus'] = Vplus
#         self.metadata['Vminus'] = Vminus
#         self.metadata['Iminus'] = Iminus
        
        self.tdmsfile = None
        self.outerindexlist = []
        self.monodfconcatenated = pd.DataFrame({})
        self.pivottable = pd.DataFrame({})
        self.pivottabledX = pd.DataFrame({})
        self.pivottabledY = pd.DataFrame({})
        self.pivottabledXdY = pd.DataFrame({})
        # initiate data
        
        self.__maskinitialized = False
        self.__positivemask = None
        self.__negativemask = None
        self.__increasemask = None
        self.__decreasemask = None
        # initiate mask
        
       
    def saveToPickles(self, appendix = ''):
        """
        save processed data to .pkl files
        appendix: comments about this save
        this function will also save the parameters for data processing in metadata.txt, such as differential & filtering
        """
        now = datetime.now()
        output_path = self.folder + '/Jupyter Output' + appendix + ' ' + now.strftime('%Y-%m-%d %H %M')
        os.mkdir(output_path)
        # create folder. folder name = 'Jupyter Output' + your comment + datetime
        
        with open(output_path + '/metadata.txt', 'w') as f:
            json.dump(self.metadata, f)
        # write metadata to txt file
        
        if len(self.outerindexlist) > 0:
            with open(output_path + '/outerindex.pkl', 'wb') as f:
                pickle.dump(self.outerindexlist, f)
        # write outer sweep channel values to .pkl file
        
        if not self.monodfconcatenated.empty:
            self.monodfconcatenated.to_pickle(output_path + '/monodfconcatenated.pkl')
        
        if not self.pivottable.empty:
            self.pivottable.to_pickle(output_path + '/pivottable.pkl')
        
        if not self.pivottabledX.empty:
            self.pivottabledX.to_pickle(output_path + '/pivottabledX.pkl')
            
        if not self.pivottabledY.empty:
            self.pivottabledY.to_pickle(output_path + '/pivottabledY.pkl')
            
        if not self.pivottabledXdY.empty:
            self.pivottabledXdY.to_pickle(output_path + '/pivottabledXdY.pkl')
        # write Pandas dataframes to .pkl files
            
    
    def loadFromPickles(self, jupyteroutputfolder = 'Jupyter Output'):
        """
        load previously saved data
        """
        latest_folder = max([self.folder + '/' + f for f in os.listdir(self.folder) if f.startswith(jupyteroutputfolder)],\
                            key=os.path.getmtime)
        path = latest_folder
        # find the path of previously saved data that starts with jupyteroutputfolder: str
        # if multiple folders satisfy the criteria, read from the latest one
        
        with open(path + '/metadata.txt', "r") as read_file:
            metadata = json.load(read_file)        
        self.metadata = metadata
        # read metadata
        
        if 'monodfconcatenated.pkl' in os.listdir(path):
            self.monodfconcatenated = pd.read_pickle(path + '/monodfconcatenated.pkl')
        
        if 'pivottable.pkl' in os.listdir(path):
            self.pivottable = pd.read_pickle(path + '/pivottable.pkl')
            
        if 'pivottabledX.pkl' in os.listdir(path):
            self.pivottabledX = pd.read_pickle(path + '/pivottabledX.pkl')
            
        if 'pivottabledY.pkl' in os.listdir(path):
            self.pivottabledY = pd.read_pickle(path + '/pivottabledY.pkl')
            
        if 'pivottabledXdY.pkl' in os.listdir(path):
            self.pivottabledXdY = pd.read_pickle(path + '/pivottabledXdY.pkl')
        # read .pkl files to Pandas dataframes
            
        if 'outerindex.pkl' in os.listdir(path):
            with open(path + '/outerindex.pkl', 'rb') as f:
                outerindexlist = pickle.load(f)
                self.outerindexlist = outerindexlist
        else:
            self.outerindexlist = self.monodfconcatenated.index.unique()  
        # read outer sweep channel values
#         self.openTdms()
        
    
    def openTdms(self):
        """
        use the TdmsFile package to open the tdms file
        if the experiment contains multiple tdms files, concatenate them into a single tdms file
        """
        if len(self.filelist) == 1:
            self.tdmsfile = TdmsFile(self.filelist[0])
        # if there is a single tdms file for this experiment, open it
        else:
            # if this experiment has multiple tdms files
            filepath = self.folder + '/' + 'concatenated.tdms'
            self.concatenateTdms(filepath)
            self.tdmsfile = TdmsFile(filepath)
            # concatenate all tdms files into one file
            # open the concatenated tdms
            
            archive_path = self.folder + '/' + 'archive separate tdms'
            if not os.path.isdir(archive_path):
                os.mkdir(archive_path)
            for filename in self.filelist:
                shutil.move(filename, archive_path)
            # move the individual tdms files into a folder called "archived"
            self.filelist = [self.folder + '/' + 'concatenated.tdms']
            # leave only the concatenated tdms, update Scan2D.filelist
            
    
    def concatenateTdms(self, filepath):
        """
        concatenate multiple tdms files into one file
        create a new tdms file, and then dump all data into this file
        """
        group_counter = 0
        # record in total how many groups have we copied
        
        with TdmsWriter(filepath) as concatenated_file:
            # create a new tdms file specified by filepath, to dump all the data
            for filename in tqdm(self.filelist):
                # iterate individual tdms files
                with TdmsFile.open(filename) as tdms_file:
                    # iterate all groups in each tdms file
                    groups = tdms_file.groups()
                    for group in groups:
                        new_group_name = 'Group{:06d}'.format(group_counter)
                        # name the new group according to the group counter
                        new_group_object = GroupObject(new_group_name, properties = group.properties)
                        # create new group object
                        for channel in group.channels():
                            # iterate all channels in each group
                            new_channel_object = ChannelObject(new_group_name, channel.name, channel[:],\
                                                               properties = channel.properties)
                            # create new channel object
                            concatenated_file.write_segment([new_group_object] + [new_channel_object])
                            # copy group + channel into the new file
                        group_counter += 1

        
    def printChannelInfo(self):
        """
        print out what channels have been recorded in the experiment
        help you determine Scan2D.metadata['dataofinterest']
        """
        groups = self.tdmsfile.groups()
        
        if len(groups) == 0:
            print('no data')
            return
        
        firstgroup = groups[0]
        
        for channel in firstgroup.channels():
            print(channel)
            
    
    def tdmsToAveragedMonoDfConcatenated(self):
        """
        convert the tdms file to a Pandas dataframe, which is indexed by the outer sweep channel
        Note:
        averaged means: when you take multiple curves at the same condition, they are averaged to get one curve
        mono means: filter out the "monotonic" part of data, for example if you had a triangle shaped sourcing 
        when taking IV curves, this function can keep the monotonic part
        df means: Pandas dataframe
        concatenated means: all data in this experiment are converted into a single dataframe
        """
        groups = self.tdmsfile.groups()
        
        index_count = len(groups) // self.metadata['groupsize']
        # index_count = how many curves are there after averaging by groupsize
        
        for i in tqdm(range(index_count)):
            groups_with_same_index = groups[self.metadata['groupsize'] * i: self.metadata['groupsize'] * (i + 1)]
            # groups_with_same_index = these data groups are taken at the same condition, and they need to be averaged
            averaged_mono_df_single = self.tdmsToAveragedMonoDfSingle(groups_with_same_index)
            # averaged_mono_df_single = a single dataframe converted from these data groups
            # it averages these groups, and monotonic part is extracted
            if i == 0:
                mono_df_concatenated = averaged_mono_df_single
            # if this is the first curve, copy it to mono_df_concatenated
            else:
                mono_df_concatenated = mono_df_concatenated.append(averaged_mono_df_single)
            # if not, append it to the existing mono_df_concatenated
        
        self.monodfconcatenated = mono_df_concatenated.set_index(self.metadata['outersweepchannel'])
        # set index of mono_df_concatenated to be outer sweep channel
        self.outerindexlist = self.monodfconcatenated.index.unique()
        # extract the values of outer sweep channel
        
        
    def tdmsToAveragedMonoDfSingle(self, group_list):
        """
        from tdmsfile, read multiple data groups in group_list, convert them to dataframes
        then average the dataframes to get a single dataframe
        then extract the monotonic part of this dataframe
        """
        df_list = [self.tdmsToDfSingle(group) for group in group_list]
        # df_list is a list of dataframes. each dataframe is read from one group
        
        numpy_array_list = [df.to_numpy() for df in df_list]
        averaged_numpy_array = sum(numpy_array_list) / len(df_list)
        # average the values of dataframes in df_list. get one numpy 2d array
        
        averaged_df = pd.DataFrame(averaged_numpy_array, columns = df_list[0].columns)
        # create a new dataframe from the averaged numpy 2d array
        
        if self.metadata['monodirection'] != 'all':
            return self.extractMonotonicPart(averaged_df)
        return averaged_df
        # extract the monotonic part
            
        
    def __calcV2TV4T(self, df):
        """
        calculate V2T and V4T from Vplus and Vminus
        """
        if self.metadata['scantype'] == 'IV':
            df['V2T'] = df[self.metadata['source']]
        # for IV measurement, the source AO is the V2T
        if self.metadata['scantype'] == 'Conductance':
            df['V2T'] = self.metadata['sourceamp']
        # for lockin conductance measurement, the source amplitude is the V2T
        
        if bool(self.metadata['Vplus']) and bool(self.metadata['Vminus']) and self.metadata['Vplus'] != self.metadata['Vminus']:
            df['V4T'] = df[self.metadata['Vplus']] - df[self.metadata['Vminus']]
        # if we have both Vplus and Vminus and they are different, V4T = Vplus - Vminus
        elif bool(self.metadata['Vplus']):
            df['V4T'] = df[self.metadata['Vplus']]
        # if Vplus and Vminus the same, differential mode, V4T = Vplus
        elif bool(self.metadata['Vminus']):
            df['V4T'] = df[self.metadata['source']] - df[self.metadata['Vminus']]
        else:
            df['V4T'] = df['V2T']
            
    
    def __calcGRforGVsgScan(self, df):
        """
        calculate R and G for lockin conductance measurement, from V4T, V2T and Iminus
        """  
        df['R2T'] = df['V2T'] / df[self.metadata['Iminus']]
        df['G2T'] = 1/ df['R2T']
        df['R4T'] = df['V4T'] / df[self.metadata['Iminus']]
        df['G4T'] = 1 / df['R4T']
        
        
    def __iniMask(self, df_single):
        """
        create boolean mask for dataframe, to extract certain part of the data
        """
        self.__positivemask = df_single[self.metadata['innersweepchannel']] > 0
        # mask for positive inner sweep channel values
        self.__negativemask = df_single[self.metadata['innersweepchannel']] < 0
        # mask for negative inner sweep channel values
        self.__increasemask = df_single[self.metadata['innersweepchannel']].diff() > 0
        # mask for increasing inner sweep channel values
        self.__decreasemask = df_single[self.metadata['innersweepchannel']].diff() < 0
        # mask for decreasing inner sweep channel values
        self.__maskinitialized = True
        
    
    def extractMonotonicPart(self, df_single):
        """
        extract the monotonic part of a single dataframe, always gives an ascending output
        Note:
        this function has a triangular or ramping inner sweep channel in mind
        this function probably can not deal with more complicated sweep shape
        """
        if not self.__maskinitialized:
            self.__iniMask(df_single)
        # initiate boolean mask
        
        if self.metadata['monodirection'] == 'monoup':
            df_monoup = df_single[self.__increasemask & self.__negativemask]
            return df_monoup.append(df_single[self.__increasemask & self.__positivemask])
        # 'monoup' means increasing portion
        # 'monoup' = <0,increase + >0,increase
     
        elif self.metadata['monodirection'] == 'monodown':
            df_monodown = df_single[self.__decreasemask & self.__positivemask]
            df_monodown = df_monodown.append(df_single[self.__decreasemask & self.__negativemask])
            return df_monodown.iloc[::-1]
        # 'monodown' means decreasing portion
        # 'monodown' = revert(>0,decrease + <0,decrease)

        elif self.metadata['monodirection'] == 'increasefrom0':
            df_increasefrom0 = df_single[self.__decreasemask & self.__negativemask]
            df_increasefrom0 = df_increasefrom0.iloc[::-1]
            return df_increasefrom0.append(df_single[self.__increasemask & self.__positivemask])
        # 'increasefrom0' means ramping away from 0
        # 'increasefrom0' = revert(<0,decrease) + >0,increase
        
        elif self.metadata['monodirection'] =='decreaseto0':
            df_decreaseto0 = df_single[self.__increasemask & self.__negativemask]
            return df_decreaseto0.append(df_single[self.__decreasemask & self.__positivemask].iloc[::-1])
        # 'decreaseto0' means ramping down to 0
        # 'decreaseto0' = <0,increase + revert(>0,decrease)
            
        else:
            return df_single
    
        
    def tdmsToDfSingle(self, group):
        """
        read data from a single group in the tdms file, convert it to a Pandas dataframe
        """
        if self.metadata['scantype'] == 'IV':
            num_of_points = group['AI1'].data.size
        elif self.metadata['scantype'] == 'Conductance':
            num_of_points = group['X1'].data.size
        # get the total number of datapoints of this curve
        
        df=pd.DataFrame({'datapoint':np.arange(num_of_points)})
        # the datapoint column stores the numeric index
        
        for column in self.metadata['dataofinterest']:
            # iterate the channels in this group
            if column in ['V2T', 'V4T', 'G4T', 'R4T', 'G2T', 'R2T']:
                df[column] = np.zeros(num_of_points)
            # will calculate V2T, V4T, etc later 
                
            data = group[column].data
            # read data from this channel
            
            if len(data) == num_of_points:
                df[column] = data
            elif column == self.metadata['outersweepchannel']:
                outerindex = data[0] + 1e-12 * random.random()
                # this small random component is to deal with the possible recurrent outer sweep channel values
                # for example the magnetic field does not change between two IV curves
                # during later processing these two IV curves will have the same index, and cause problem
                df[column] = [outerindex for _ in range(num_of_points)]
            else:
                df[column] = [data[0] for _ in range(num_of_points)]
                # sometimes there is a single value in a channel, we need to duplicate it to match the dataframe size
        
        if self.metadata['convertfilename']:
            self.convertFilenameToIndex(group.name, df)
            # convert group index to outer sweep channel value if outer sweep channel is not recorded
        
        self.__calcV2TV4T(df)
        # calculate V2T and V4T, add them to dataframe columns
        if self.metadata['scantype'] == 'Conductance':
            self.__calcGRforGVsgScan(df)
        # for lockin measurements, calculate conductance and resistance, add them to dataframe columns
        return df
    

    def convertFilenameToIndex(self, groupname, df_single):
        """
        convert a groupname to a outer sweep channel index
        """
        # find all numbers in the string using re.findall()
        numbers = re.findall(r'\d+', groupname)
        # join the numbers into one substring
        numeric_substring = ''.join(numbers)
        # return the numeric substring
        outerindex = (int(numeric_substring) // self.metadata['groupsize']) * self.metadata['convertparam']['step'] +\
                    self.metadata['convertparam']['start']
        df_single[self.metadata['outersweepchannel']] = outerindex

    
    def regroup(self, by, num_of_index = 1, criterion = 'value'):
        self.metadata['regroup_by'] = by
        self.monodfconcatenated.reset_index(inplace = True)
        
        if criterion == 'index':
            self.monodfconcatenated['label'] = [0] * len(self.monodfconcatenated)
            group_length = len(self.monodfconcatenated) // num_of_index
            
            for i in range(num_of_index):
                
                self.monodfconcatenated.loc[i * group_length : (i + 1) * group_length, 'label']=                np.mean(self.monodfconcatenated.loc[i * group_length : (i + 1) * group_length, by])
        
        elif criterion == 'value':
            print('assigning scan2D labels')
            self.monodfconcatenated.loc[::, 'label'] = self.monodfconcatenated[by].values
        
        elif criterion == 'kmeans':
            X = self.monodfconcatenated[[by]].values
            
            kmeans = KMeans(n_clusters = num_of_index)
            kmeans.fit(X)
            self.monodfconcatenated['cluster'] = kmeans.labels_
            
            mappingToRealValue = lambda y : np.mean(self.monodfconcatenated[self.monodfconcatenated['cluster'] == y][by])
            self.monodfconcatenated.loc[::, 'label'] = list(map(mappingToRealValue, self.monodfconcatenated['cluster'].values))
                        
        self.monodfconcatenated.set_index('label', inplace = True)
        self.outerindexlist = self.monodfconcatenated.index.unique()
        
    
    def interpXY(self, xdata, ydata, xtarget, num_of_points):
        ylist = np.empty(num_of_points)
        ylist.fill(np.nan)
        
        xmin_local = xdata.min()
        xmax_local = xdata.max()
        
        for i, x in enumerate(xtarget):
            if x < xmin_local or x > xmax_local:
                continue
            ylist[i] = np.interp(x, xdata, ydata)

        return pd.Series(ylist, index = xtarget)
    
    
    def createPivotTable(self, parameters = dict(x = 'AI1', y = 'V4T', num_of_points = 1000,                                                 auto_xrange = True, xmin = -1e-7, xmax = 1e-7,                                                 lowpass_filter = False, cutoff = 60, fs = 1000,                                                 remove_repeated_x = False)):
        x = parameters['x']
        y = parameters['y']
        num_of_points = parameters['num_of_points']
        lowpass_filter = parameters['lowpass_filter']
        cutoff = parameters['cutoff']
        fs = parameters['fs']
        remove_repeated_x = parameters['remove_repeated_x']
        auto_xrange = parameters['auto_xrange']
        xmin = parameters['xmin']
        xmax = parameters['xmax']
        
        self.metadata['pivottablecolumn'] = x
        self.metadata['pivottablerow'] = self.metadata['outersweepchannel'] if 'regroup_by' not in self.metadata else self.metadata['regroup_by']
        self.metadata['pivottablevalue'] = y
        
        if auto_xrange:
            xmin = min(self.monodfconcatenated[x])
            xmax = max(self.monodfconcatenated[x])
        xtarget = np.linspace(xmin, xmax, num_of_points)
        
        series_list = []
        
        for outerindex in tqdm(self.outerindexlist):
            xdata = self.monodfconcatenated.loc[outerindex, x]
            ydata = self.monodfconcatenated.loc[outerindex, y]
            
            if remove_repeated_x:
                xdata, ydata = IVProcessandPlot.removeRepeatedPoints(xdata, ydata)
            
            if lowpass_filter:
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
                xdata = signal.filtfilt(b, a, xdata)
                ydata = signal.filtfilt(b, a, ydata)
                self.metadata['pivottablelowpassfiltered'] = True

            series_list.append(self.interpXY(xdata, ydata, xtarget, num_of_points))
        
        self.pivottable = pd.concat(series_list, keys = self.outerindexlist).unstack(level = -1)
        
   
    def calcSeriesDVDI(self, series, diff_step = 10, inter = 1):
        length = len(series)
        step = diff_step if diff_step >= 1 else diff_step * length
        
        index_list = []
        dvdi_list = []
        
        for i in range(0, length - step, inter):
            start = i
            end = i + step
            mid = i + step // 2
            
            index_list.append(series.index[mid])
            if series.values[start] == np.nan or series.values[end] == np.nan:
                dvdi_list.append(np.nan)
                continue
            
            dvdi = np.polyfit(series.index[start: end], series.values[start: end], 1)[0]
            dvdi_list.append(dvdi)
        
        return pd.Series(dvdi_list, index = index_list)
    
    
    def differentiateTable(self, parameters = dict(columnstep = 10, columninter = 1,                                                   rowstep = 6, rowinter = 1)):
        columnstep = parameters['columnstep']
        columninter = parameters['columninter']
        rowstep = parameters['rowstep']
        rowinter = parameters['rowinter']
        
        if self.pivottable.empty:
            print('create pivot table first')
            return
        
        self.metadata['dzdx_columnstep'] = columnstep
        self.metadata['dzdy_rowstep'] = rowstep
        
        dx_series_list = []
        dy_series_list = []
        dydx_series_list = []
        
        if columnstep > 0:
            for index in tqdm(self.outerindexlist):
                dx_series_list.append(self.calcSeriesDVDI(self.pivottable.loc[index],                                      columnstep, columninter))
            self.pivottabledX = pd.concat(dx_series_list, keys = self.outerindexlist).                              unstack(level = -1)
        else:
            self.pivottabledX = pd.DataFrame({})
            self.pivottabledXdY = pd.DataFrame({})
        
        if rowstep > 0:
            for column in tqdm(self.pivottable.columns):
                dy_series_list.append(self.calcSeriesDVDI(self.pivottable[column],                                                          rowstep, rowinter))
            self.pivottabledY = pd.concat(dy_series_list, keys = self.pivottable.columns).                              unstack(level = -1).T
        else:
            self.pivottabledY = pd.DataFrame({})
            self.pivottabledXdY = pd.DataFrame({})
        
        if columnstep > 0 and rowstep > 0:
            for column in tqdm(self.pivottabledX.columns):
                dydx_series_list.append(self.calcSeriesDVDI(self.pivottabledX[column],                                        rowstep, rowinter))
            self.pivottabledXdY = pd.concat(dydx_series_list, keys = self.pivottabledX.columns).                                unstack(level = -1).T                           


# In[160]:


class Scan3D():
    def __init__(self, folder = '03-Sweep T B IV', 
                 metadata = dict(scantype = 'IV', groupsize = 1, monodirection = 'monoup',\
                 convertfilename = False, convertparam = dict(start = 0, step = 1),\
                 dataofinterest = ['Magnet', 'Temperature', 'AO1', 'AI2', 'AI3', 'AI4'],\
                 outmostsweepchannel = 'Temperature', outersweepchannel = 'Magnet', innersweepchannel = 'AO1',\
                 source = 'AO1', sourceamp = 0.001, Vplus = 'AI4', Vminus = 'AI3', Iminus = 'AI2')):
        
        self.folder = folder     
        self.metadata = metadata
        
        self.scanall = Scan2D(folder, metadata)
        self.outmostindexlist = []
        self.scan2Dlist = []
        
    def openTdms(self):
        self.scanall.openTdms()
        
    def tdmsToAveragedMonoDfConcatenated(self):
        self.scanall.tdmsToAveragedMonoDfConcatenated()
#       self.scanall.saveToPickles('ALL DATA')
        
    def splitDataByOutmostIndex(self, num_of_outmostindex = 1, criterion = 'value'):
        self.scanall.regroup(by = self.metadata['outmostsweepchannel'], 
                             num_of_index = num_of_outmostindex, 
                             criterion = criterion)
        self.outmostindexlist = self.scanall.outerindexlist
        
        print('assigning scan2D')
        self.scan2Dlist = []
        
        for outmostindex in self.outmostindexlist:
            scan2D = Scan2D(self.folder, self.metadata.copy())
            scan2D.monodfconcatenated = self.scanall.monodfconcatenated.loc[outmostindex].copy()
            
            scan2D.regroup(by = self.metadata['outersweepchannel'], 
                                              criterion = 'value')
            self.scan2Dlist.append(scan2D)
            
        zipped = zip(self.outmostindexlist, self.scan2Dlist)
        zipped = sorted(zipped)
        self.outmostindexlist, self.scan2Dlist = zip(*zipped)
        
        
    def saveToPickles(self):
        self.scanall.saveToPickles(appendix = '-alldata')
        
        for i, scan2D in enumerate(self.scan2Dlist):
            scan2D.saveToPickles(appendix = '-' + self.metadata['outmostsweepchannel'] + '-' + str(self.outmostindexlist[i]))
            
    def loadFromPickles(self):
        self.scan2Dlist = []
        self.scanall.loadFromPickles(jupyteroutputfolder = 'Jupyter Output-alldata')
        self.metadata = self.scanall.metadata
        self.outmostindexlist = self.scanall.outerindexlist
        
#         filelist = [f for f in os.listdir(path) if f.startswith('Jupyter Output-' + self.metadata['outmostsweepchannel'])]
        
        for index in self.outmostindexlist:
            folder = 'Jupyter Output-' + self.metadata['outmostsweepchannel'] + '-' + str(index)
            scan2D = Scan2D(self.folder, self.metadata.copy())
            scan2D.loadFromPickles(folder)
            self.scan2Dlist.append(scan2D)
            
    def createPivotTable(self, parameters = dict(x = 'AI1', y = 'V4T', num_of_points = 1000,                         lowpass_filter = False, cutoff = 60, fs = 1000,                         remove_repeated_x = False)):
        for scan2D in self.scan2Dlist:
            scan2D.createPivotTable(parameters)
            
    def differentiateTable(self, parameters = dict(columnstep = 10, columninter = 1,                                               rowstep = 6, rowinter = 1)):
        for scan2D in self.scan2Dlist:
            scan2D.differentiateTable(parameters)


# In[195]:


class ProcessandPlot():
    def removeRepeatedPoints(xdata, ydata):
        n = len(xdata)
        mid = n // 2
        
        xdata.reset_index(drop = True, inplace = True)
        ydata.reset_index(drop = True, inplace = True)
        
        if xdata.iloc[-1] < xdata.iloc[0]:
            xdata, ydata = xdata.iloc[::-1], ydata.iloc[::-1]
        
        mask = pd.Series([True] * n, index = xdata.index)
        right_max = float('-inf')
        left_min = float('inf')
        
        for right in range(mid + 1, n):
            if xdata[right] < right_max:
                mask[right] = False
            right_max = max(right_max, xdata[right])
        
        for left in range(mid, 0, -1):
            if xdata[left] > left_min:
                mask[left] = False
            left_min = min(left_min, xdata[left])
        
        return xdata[mask], ydata[mask]
    
    def extractIcVoltageThreshold(VvsI, threshold):
        Icplus = np.abs(VvsI - threshold).idxmin()
        Icminus = np.abs(VvsI + threshold).idxmin()
        return Icplus, Icminus
    
    def extractIcResistanceThreshold(dVdIvsI, threshold):
        Icplus = np.abs(dVdIvsI[dVdIvsI.index > 0] - threshold).idxmin()
        Icminus = np.abs(dVdIvsI[dVdIvsI.index < 0] - threshold).idxmin()
        return Icplus, Icminus
    
    def extractIcCoherencePeak(dVdIvsI):
        Icplus = dVdIvsI[dVdIvsI.index > 0].idxmax()
        Icminus = dVdIvsI[dVdIvsI.index < 0].idxmax()
        return Icplus, Icminus
    
    def extractIc(VI_table, dVdI_table, algorithm, threshold):
        Icplus_list = []
        Icminus_list = []
        
        for index in tqdm(VI_table.index):
            VvsI = VI_table.loc[index]
            dVdIvsI = dVdI_table.loc[index]
            
            if algorithm == 'CoherencePeak':
                Icplus, Icminus = IVProcessandPlot.extractIcCoherencePeak(dVdIvsI)
                
            elif algorithm == 'ResistanceThreshold':
                Icplus, Icminus = IVProcessandPlot.extractIcResistanceThreshold(dVdIvsI, threshold)
            
            elif algorithm == 'VoltageThreshold':
                Icplus, Icminus = IVProcessandPlot.extractIcVoltageThreshold(dVdIvsI, threshold)
            
            Icplus_list.append(Icplus)
            Icminus_list.append(Icminus)
            
        return pd.Series(Icplus_list, index = VI_table.index),               pd.Series(Icminus_list, index = VI_table.index)
   
    
    def intensityPlot(pivottable, zmin, zmax, xlabel = '', ylabel = '', zlabel = '', title = 'intensity',                          xscale = 1, yscale = 1, zscale = 1, width = 850, height = 700,                          colorscale = 'plasma', savefig = False):
        fig_intensity = go.Figure(data = go.Heatmap(z = pivottable.iloc[::,1:-1].to_numpy() * zscale,
                                              x = np.array(pivottable.columns[1:-1]) * xscale,
                                              y = pivottable.iloc[::].index * yscale,
                                              colorbar = dict(title = zlabel),
                                              zmax = zmax,
                                              zmin = zmin,
                                              colorscale = colorscale))
        fig_intensity.update_layout(
            autosize = False,
            width = width,
            height = height,
            title = title,
            xaxis = dict(
                nticks = 11,
                title = xlabel,
                ticks = 'outside'),
            yaxis = dict(
                nticks = 11,
                title = ylabel,
                ticks = 'outside'),
            font = dict(
                family = "Courier New, monospace",
                size = 18))
#         fig_intensity.show()
        if savefig:
            fig_intensity.write_image(title + ".png", width = width, height = height)
            
        return fig_intensity
    
    def waterfallPlotHorizontal(pivottable, start = 0, step = 1, num = 2, vert_offset = 0, hor_offset = 0,                                xscale = 1, yscale = 1, legendscale = 1, legendunit = '',                                xlabel = '', ylabel = '', title = 'waterfall', savefig = False):
        fig, ax = plt.subplots()
        
        for i in range(0, step * num, step):
            row = pivottable.iloc[start + i]
            index = pivottable.index[start + i]
            legend = index * legendscale
            ax.plot((row.index + hor_offset * i / step) * xscale, (row.values + vert_offset * i / step) * yscale,                     lw = .8, alpha = .7, color = [max(0, 1 - 2 * i / step / num), 0, max(2 * i / step / num - 1, 0)],                     label = np.format_float_positional(legend, precision = 5,                                                        unique = False, fractional = False,                                                        trim='k') + ' ' + legendunit)
        
        ax.legend(fontsize = 8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if savefig:
            fig.savefig(title + '.png')
        
        return fig, ax
    
    def waterfallPlotVertical(pivottable, start = 0, step = 1, num = 2, vert_offset = 0, hor_offset = 0,                              xscale = 1, yscale = 1, legendscale = 1, legendunit = '',                              xlabel = '', ylabel = '', title = '', savefig = False):
        return IVProcessandPlot.waterfallPlotHorizontal(pivottable.T, start, step, num, vert_offset, hor_offset,                                       xscale, yscale, legendscale, legendunit,                                       xlabel, ylabel, title, savefig)