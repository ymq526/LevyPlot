#!/usr/bin/env python
# coding: utf-8

# In[00]:
"""
Python module for reading, processing and plotting Levylab tdms files
"""


# Prerequisite Packages:
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
# %matplotlib inline
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
import time
from sklearn.cluster import KMeans


# In[01]:


class Scan2D():
    """
    Scan2D class, suitable for processing 1D sweep and 2D sweep data
    1D sweep example: R vs T, R vs B, a single IV curve, conductance vs time...
    2D sweep examples: IV vs B, IV vs T, G vs B vs Vsg, R vs B vs T...
    """
    def __init__(self, folder = '03-Sweep B IV',\
                 metadata = dict(scantype = 'IV', groupsize = 1, monodirection = 'monotonic',\
                                 convertfilename = False, convertparam = dict(start = 0, step = 1),\
                                 dataofinterest = ['Magnet', 'Temperature', 'AO1', 'AI2', 'AI3', 'AI4'],\
                                 outersweepchannel = 'Magnet', innersweepchannel = 'AO1',\
                                 source = 'AO1', sourceamp = 0.001, Vplus = 'AI4', Vminus = 'AI3', Iminus = 'AI2')):
        """
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
        # initiate data
        self.outerindexlist = []
        self.monodfconcatenated = pd.DataFrame({})
        # Scan2D.monodfconcatenated is the Pandas dataframe that contains all data we care about
        # all subsequent data processing is based on this dataframe

        self.pivottable = pd.DataFrame({})
        self.pivottabledX = pd.DataFrame({})
        self.pivottabledY = pd.DataFrame({})
        self.pivottabledXdY = pd.DataFrame({})
        # these pivot tables are used to store the interpolated data for intensity plots
        
        self.__maskinitialized = False
        self.__positivemask = None
        self.__negativemask = None
        self.__increasemask = None
        self.__decreasemask = None
        # initiate mask


    def append(self, experiment):
        """
        append another scan2D object to the current scan2D object
        by appending the monodfconcatenated and outerindexlist
        """
        self.outerindexlist = self.outerindexlist.append(experiment.outerindexlist)
        self.monodfconncatenated = self.monodfconcatenated.append(experiment.monodfconcatenated)
        print('Scan2D append finished. Recreate pivottable if needed')
        
       
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
        
        print("Concatenating Tdms Files")
        time.sleep(0.5)
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
        
        print("Reading from TdmsFile to Pandas Dataframe")
        time.sleep(0.5)
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
        # Scan2D.monodfconcatenated is the Pandas dataframe that contains all data we care about
        # all subsequent data processing is based on this dataframe
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
            return df_monoup.iloc[1::].append(df_single[self.__increasemask & self.__positivemask])
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

    
    def regroup(self, by, num_of_index = 1, criterion = 'value', thresholds = []):
        """
        set the index of Scan2D.monodfconcatenated to another column
        this function creates a "label" column that classifies the data, and then set index to "label"
        """
        self.metadata['regroup_by'] = by
        # record our action in Scan2D.metadata
        self.monodfconcatenated.reset_index(inplace = True)
        # reset index

        if criterion == 'index':
            # classify the data by their sequence
            # suitable for the case when each curve has the same amount of datapoints
            self.monodfconcatenated['label'] = [0] * len(self.monodfconcatenated)
            # initialize label column
            group_length = len(self.monodfconcatenated) // num_of_index
            for i in range(num_of_index):
                self.monodfconcatenated.loc[i * group_length : (i + 1) * group_length, 'label'] =\
                    np.mean(self.monodfconcatenated.loc[i * group_length : (i + 1) * group_length, by])
        
        elif criterion == 'value':
            # classify the data by values of "by" column
            # suitable for the case when each curve was taken with a fixed outer sweep channel value
            self.monodfconcatenated.loc[::, 'label'] = self.monodfconcatenated[by].values
        
        elif criterion == 'kmeans':
            # classify the data with sklearn.kmeans classifier
            # suitable for the case when outer sweep channel is fluctuating during one curve
            # at the same time each curve does not has the same num of points
            X = self.monodfconcatenated[[by]].values
            
            kmeans = KMeans(n_clusters = num_of_index)
            kmeans.fit(X)
            self.monodfconcatenated['cluster'] = kmeans.labels_
            
            mapping = {}
            for y in self.monodfconcatenated['cluster'].unique():
                mapping[y] = np.mean(self.monodfconcatenated[self.monodfconcatenated['cluster'] == y][by])
            
            mappingToRealValue = lambda y: mapping[y]
            self.monodfconcatenated.loc[::, 'label'] = list(map(mappingToRealValue, self.monodfconcatenated['cluster'].values))
            # the kmeans classifier will return the labeling 0, 1, 2, 3...
            # this step is to map the kmeans labeling to actual averaged value of the channel you specify

        elif criterion == 'threshold':
            # classify the data with manually specified thresholds of the outmostsweep channel
            # can only be used to split scan3D.scanall
            self.monodfconcatenated['label'] = [0] * len(self.monodfconcatenated)
            # initialize label column and cluster column
            if len(thresholds) != num_of_index:
                print('conflit between set num_of_index and len of thresholds')
            
            for i, threshold in enumerate(thresholds):
                lowerlimit = threshold[0]
                upperlimit = threshold[1]
                index_within_threshold = self.monodfconcatenated[(self.monodfconcatenated[by] > lowerlimit)\
                                                                  & (self.monodfconcatenated[by] < upperlimit)].index
                averaged_index = np.mean(self.monodfconcatenated.loc[index_within_threshold, by])
                self.monodfconcatenated.loc[index_within_threshold, 'label'] = averaged_index
                        
        self.monodfconcatenated.set_index('label', inplace = True)
        if criterion == 'index' or criterion == 'kmeans':
            self.outerindexlist = self.monodfconcatenated.index.unique()[0:num_of_index]
        elif criterion == 'value' or criterion == 'threshold':
            self.outerindexlist = self.monodfconcatenated.index.unique()
        # set index to "label", record the unique values of the outer sweep channel
        
    
    def interpXY(self, xdata, ydata, xtarget, num_of_points):
        """
        given a curve with x and y data, interpolate x to xtarget
        return a Pandas series
        """
        ylist = np.empty(num_of_points)
        ylist.fill(np.nan)
        # initialize target y, filled with np.nan
        
        xmin_local = xdata.min()
        xmax_local = xdata.max()
        
        for i, x in enumerate(xtarget):
            if x < xmin_local or x > xmax_local:
                continue
            # if the target x is out of the range of this curve, ignore
            ylist[i] = np.interp(x, xdata, ydata)
            # interpolation

        return pd.Series(ylist, index = xtarget)
        # return Pandas series, index = xtarget and value = interpolated y
    
    
    def createPivotTable(self, parameters = dict(x = 'AI1', y = 'V4T', num_of_points = 1000,\
                                                 auto_xrange = True, xmin = -1e-7, xmax = 1e-7,\
                                                 lowpass_filter = False, cutoff = 60, fs = 1000,\
                                                 remove_repeated_x = False)):
        """
        from Scan2D.monodfconcatenated, create a pivot table that stores a certain channel
        vs outer sweep channel vs inner sweep channel
        the main purpose of this function is to clean the data for intensity plots
        """
        x = parameters['x']
        y = parameters['y']
        # pivottable row = outer sweep channel
        # pivottable column = x you specify
        # pivottable value = y you specify
        # each pivottable row = y vs x at certain outer sweeo channel value
        num_of_points = parameters['num_of_points']
        lowpass_filter = parameters['lowpass_filter']
        cutoff = parameters['cutoff']
        # cutoff freq of low pass filter, usually 60Hz
        fs = parameters['fs']
        # for low pass filtering, usually fs = output rate of lockin
        remove_repeated_x = parameters['remove_repeated_x']
        auto_xrange = parameters['auto_xrange']
        xmin = parameters['xmin']
        xmax = parameters['xmax']
        
        self.metadata['pivottablecolumn'] = x
        self.metadata['pivottablerow'] = self.metadata['outersweepchannel'] if 'regroup_by' not in self.metadata else self.metadata['regroup_by']
        self.metadata['pivottablevalue'] = y
        # record the row, column and value channels of the pivottable to Scan2D.metadata
        
        if auto_xrange:
            xmin = min(self.monodfconcatenated[x])
            xmax = max(self.monodfconcatenated[x])
        xtarget = np.linspace(xmin, xmax, num_of_points)
        # specify the x target to map the curves onto
        
        series_list = []
        # initialize a list to store Pandas series, each series will represent an interpolated curve

        print("Building Pivot Table")
        time.sleep(0.5)
        for outerindex in tqdm(self.outerindexlist):
            # iterate each curve in Scan2D.monodfconcatenated
            # xydata = self.monodfconcatenated.loc[outerindex, [x, y]].iloc[1:-1:]
            # xydata.sort_values(by = x, inplace = True)
            # xdata = xydata.loc[::, x]
            # ydata = xydata.loc[::, y]
            xdata = self.monodfconcatenated.loc[outerindex, x].iloc[1:-1:]
            ydata = self.monodfconcatenated.loc[outerindex, y].iloc[1:-1:]
            # extract x and y data from Pandas series
            if remove_repeated_x:
                xdata, ydata = ProcessandPlot.removeRepeatedPoints(xdata, ydata)
            # this step is to remove the non-monotonic part of xdata,
            # especially to deal with the case when current suddenly drops near a superconducting transition
            if lowpass_filter:
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
                xdata = signal.filtfilt(b, a, xdata)
                ydata = signal.filtfilt(b, a, ydata)
                self.metadata['pivottablelowpassfiltered'] = True
                # low pass filtering to filter out the >60Hz signal

            series_list.append(self.interpXY(xdata, ydata, xtarget, num_of_points))
        
        self.pivottable = pd.concat(series_list, keys = self.outerindexlist).unstack(level = -1)
        # concatenate all series and unstack, to get the pivottable
        self.pivottable.sort_index(axis = 0, inplace = True, ascending=True)
        # sort the pivottable by index
        
   
    def calcSeriesDVDI(self, series, diff_step = 10, inter = 1):
        """
        differentiate Pandas series
        diff_step = the step size for differentiation
        inter = output rate, for example if inter = 1, output every point, if inter = 2, output one per two points
        """
        length = len(series)
        step = diff_step if diff_step >= 1 else diff_step * length
        # if diff step < 1, we instead use diff_step as a percentage
        
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
            # get the slope of linearfit
        
        return pd.Series(dvdi_list, index = index_list)
        # return Pandas series, index = original index, value = d value / d index
    
    
    def differentiateTable(self, parameters = dict(columnstep = 10, columninter = 1, rowstep = 6, rowinter = 1)):
        """
        differentiate a pivottable, useful for getting differential resistance or transconductance
        columnstep = differentiation stepsize along column direction
        rowstep = differentiation stepsize along row direction
        if we don't want to differentiate along a certain direction, set its step = 0
        """
        columnstep = parameters['columnstep']
        columninter = parameters['columninter']
        rowstep = parameters['rowstep']
        rowinter = parameters['rowinter']
        
        if self.pivottable.empty:
            print('create pivot table first')
            return
        # if the pivottable has not been created, need to create a pivottable before differentiation
        
        self.metadata['dzdx_columnstep'] = columnstep
        self.metadata['dzdy_rowstep'] = rowstep
        # record the differentiation parameters in Scan2D.metadata
        
        dx_series_list = []
        dy_series_list = []
        dydx_series_list = []
        
        if columnstep > 0:
            print("Differentiating Pivot Table against Column")
            time.sleep(0.5)
            # differentiate against column
            # iterate rows, differentiate each row, and concatenate then unstack
            for index in tqdm(self.pivottable.index):
                dx_series_list.append(self.calcSeriesDVDI(self.pivottable.loc[index], columnstep, columninter))
            self.pivottabledX = pd.concat(dx_series_list, keys = self.pivottable.index).unstack(level = -1)
        else:
            self.pivottabledX = pd.DataFrame({})
            self.pivottabledXdY = pd.DataFrame({})

        if rowstep > 0:
            print("Differentiating Pivot Table against Row")
            time.sleep(0.5)
            # differentiate against row
            # iterate columns, differentiate each column, and concatenate then unstack then transpose
            for column in tqdm(self.pivottable.columns):
                dy_series_list.append(self.calcSeriesDVDI(self.pivottable[column], rowstep, rowinter))
            self.pivottabledY = pd.concat(dy_series_list, keys = self.pivottable.columns).unstack(level = -1).T
        else:
            self.pivottabledY = pd.DataFrame({})
            self.pivottabledXdY = pd.DataFrame({})
    
        if columnstep > 0 and rowstep > 0:
            print("Differentiating Pivot Table against Both Row and Column")
            time.sleep(0.5)
            # differentiate against row after differentiating against column
            for column in tqdm(self.pivottabledX.columns):
                dydx_series_list.append(self.calcSeriesDVDI(self.pivottabledX[column], rowstep, rowinter))
            self.pivottabledXdY = pd.concat(dydx_series_list, keys = self.pivottabledX.columns).unstack(level = -1).T                           


# In[02]:


class Scan3D():
    """
    Scan3D class, suitable for processing 3D sweep data
    It processes a 3D sweep with a list of 2D sweep, indexed by the outmost sweep channel
    #D sweep example: IV vs B vs T, IV vs B vs Vbg, R vs B vs T vs Vbg...
    """
    def __init__(self, folder = '03-Sweep T B IV', 
                 metadata = dict(scantype = 'IV', groupsize = 1, monodirection = 'monoup',\
                 convertfilename = False, convertparam = dict(start = 0, step = 1),\
                 dataofinterest = ['Magnet', 'Temperature', 'AO1', 'AI2', 'AI3', 'AI4'],\
                 outmostsweepchannel = 'Temperature', outersweepchannel = 'Magnet', innersweepchannel = 'AO1',\
                 source = 'AO1', sourceamp = 0.001, Vplus = 'AI4', Vminus = 'AI3', Iminus = 'AI2')):
        """
        initialization. metadata of Scan3D is the metadata of Scan2D + outmost sweep channel info
        """
        self.folder = folder     
        self.metadata = metadata
        
        self.scanall = Scan2D(folder, metadata)
        # Scan3D.scanall is a Scan2D object, a Pandas dataframe that stores ALL data from this experiment
        self.outmostindexlist = []
        # Scan3D.outmostindexlist stores the values that the outmost sweep channel can take
        self.scan2Dlist = []
        # Scan3D.scan2Dlist stores the Scan2D objects in a list, that are divided by their outmost sweep channel value
        
    def openTdms(self):
        """
        open the tdms object in Scan3D.scanall, for reading data
        """
        self.scanall.openTdms()
        
    def tdmsToAveragedMonoDfConcatenated(self):
        """
        read ALL data to Scan3D.scanall.monodfconcatenated
        """
        self.scanall.tdmsToAveragedMonoDfConcatenated()
#       self.scanall.saveToPickles('ALL DATA')
        

    def sortByOutmostIndex(self):
        """
        sort Scan3D.scan2Dlist by the outmost sweep channel index
        """
        zipped = zip(self.outmostindexlist, self.scan2Dlist)
        zipped = sorted(zipped)
        self.outmostindexlist, self.scan2Dlist = zip(*zipped)


    def splitDataByOutmostIndex(self, num_of_outmostindex = 1, criterion = 'value', thresholds = []):
        """
        split the data stored in Scan3D.scanall into many parts, according to the outmost sweep channel value
        """
        self.scanall.regroup(by = self.metadata['outmostsweepchannel'], 
                             num_of_index = num_of_outmostindex, 
                             criterion = criterion,
                             thresholds = thresholds)
        self.outmostindexlist = self.scanall.outerindexlist
        self.scan2Dlist = []
        
        print('Dividing Scanall into a list of Scan2D Objects')
        time.sleep(0.5)
        for outmostindex in tqdm(self.outmostindexlist):
            scan2D = Scan2D(self.folder, self.metadata.copy())
            scan2D.monodfconcatenated = self.scanall.monodfconcatenated.loc[outmostindex].copy()
            # create new Scan2D object, and assign part of scanall data to the Scan2D object
            
            scan2D.regroup(by = self.metadata['outersweepchannel'], criterion = 'value')
            self.scan2Dlist.append(scan2D)
            # regoup the Scan2D object by outer sweep channel values,
            # and then append the Scan2D object to Scan3D.scan2Dlist
        self.sortByOutmostIndex()


    def append(self, experiment):
        """
        append another scan3D object to the current scan3D object
        by appending scanall, outmostindexlist and scan2Dlist
        """
        self.scanall.append(experiment.scanall)
        self.outmostindexlist = self.outmostindexlist + experiment.outmostindexlist
        self.scan2Dlist = self.scan2Dlist + experiment.scan2Dlist
        self.sortByOutmostIndex()

        
    def saveToPickles(self):
        """
        save the Scan3D object, by saving Scan3D.scanall first, 
        then saving each of the Scan2D object in the Scan3D.scan2Dlist
        """
        self.scanall.saveToPickles(appendix = '-alldata')
        # save Scan3D.scanall, appendix is "-alldata"
        
        for i, scan2D in enumerate(self.scan2Dlist):
            scan2D.saveToPickles(appendix = '-' + self.metadata['outmostsweepchannel'] + '-' + str(self.outmostindexlist[i]))
            # save each Scan2D in scan2Dlist, appendix is "-channel-value" for example "-T-0.050"

    def loadFromPickles(self):
        """
        load the Scan3D object from previously saved .pkl files, by loading Scan3D.scanall first,
        then loading the Scan2D objects one by one, and append them to the Scan3D.scan2Dlist
        """
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
            
    def createPivotTable(self, parameters = dict(x = 'AI1', y = 'V4T', num_of_points = 1000,\
                                                 lowpass_filter = False, cutoff = 60, fs = 1000, remove_repeated_x = False)):
        """
        create pivot table for each Scan2D object in Scan3D.scan2Dlist
        """
        for scan2D in self.scan2Dlist:
            scan2D.createPivotTable(parameters)
            
    def differentiateTable(self, parameters = dict(columnstep = 10, columninter = 1, rowstep = 6, rowinter = 1)):
        """
        differentiate pivot table for each Scan2D object in Scan3D.scan2Dlist
        """
        for scan2D in self.scan2Dlist:
            scan2D.differentiateTable(parameters)


# In[03]:


class ProcessandPlot():
    """
    a class for some additional data processing and also data visualization
    """
    def removeRepeatedPoints(xdata, ydata):
        """
        remove repeated points when creating pivot table in Scan2D object
        """
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
        """
        extract critical current from voltage threshold
        """
        Icplus = np.abs(VvsI - threshold).idxmin()
        Icminus = np.abs(VvsI + threshold).idxmin()
        return Icplus, Icminus
    
    def extractIcResistanceThreshold(dVdIvsI, threshold):
        """
        extract critical current from differential resistance threshold
        """
        Icplus = np.abs(dVdIvsI[dVdIvsI.index > 0] - threshold).idxmin()
        Icminus = np.abs(dVdIvsI[dVdIvsI.index < 0] - threshold).idxmin()
        return Icplus, Icminus
    
    def extractIcCoherencePeak(dVdIvsI):
        """
        extract critical current from coherence peak (maximum dVdI)
        """
        Icplus = dVdIvsI[dVdIvsI.index > 0].idxmax()
        Icminus = dVdIvsI[dVdIvsI.index < 0].idxmax()
        return Icplus, Icminus
    
    def extractIc(VI_table, dVdI_table, algorithm, threshold):
        """
        extract critical current from Scan2D.pivottable, with the algorithm you specify
        extract both the critical current of the positive current and negative current
        return two Pandas series, index = outer sweep channel, value = positive Ic or negative Ic
        """
        Icplus_list = []
        Icminus_list = []
        
        for index in tqdm(VI_table.index):
            VvsI = VI_table.loc[index]
            dVdIvsI = dVdI_table.loc[index]
            
            if algorithm == 'CoherencePeak':
                Icplus, Icminus = ProcessandPlot.extractIcCoherencePeak(dVdIvsI)
                
            elif algorithm == 'ResistanceThreshold':
                Icplus, Icminus = ProcessandPlot.extractIcResistanceThreshold(dVdIvsI, threshold)
            
            elif algorithm == 'VoltageThreshold':
                Icplus, Icminus = ProcessandPlot.extractIcVoltageThreshold(dVdIvsI, threshold)
            
            Icplus_list.append(Icplus)
            Icminus_list.append(Icminus)
            
        return pd.Series(Icplus_list, index = VI_table.index), pd.Series(Icminus_list, index = VI_table.index)
   
    
    def intensityPlot(pivottable, zmin, zmax, xlabel = '', ylabel = '', zlabel = '', title = 'intensity',\
                      xscale = 1, yscale = 1, zscale = 1, width = 850, height = 700,\
                      colorscale = 'plasma', savefig = False):
        """
        intensity plot of Scan2D.pivottable
        Note:
        need to manually specify the colorbar range zmin and zmax
        """
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
    
    def waterfallPlotHorizontal(pivottable, start = 0, step = 1, num = 2, vert_offset = 0, hor_offset = 0,\
                                xscale = 1, yscale = 1, legendscale = 1, legendunit = '',\
                                xlabel = '', ylabel = '', title = 'waterfall', savefig = False):
        """
        plot the horizontal linecuts of Scan2D.pivottable
        start = starting index
        step = index step
        num = how many curves in total
        vert_offset and hor_offset = offset in vertical or horizontal direction between nearest curves
        """
        fig, ax = plt.subplots()
        
        for i in range(0, step * num, step):
            row = pivottable.iloc[start + i]
            index = pivottable.index[start + i]
            legend = index * legendscale
            ax.plot((row.index + hor_offset * i / step) * xscale, (row.values + vert_offset * i / step) * yscale,\
                     lw = .8, alpha = .7, color = [max(0, 1 - 2 * i / step / num), 0, max(2 * i / step / num - 1, 0)],\
                     label = np.format_float_positional(legend, precision = 5,\
                                                        unique = False, fractional = False,\
                                                        trim='k') + ' ' + legendunit)
        
        ax.legend(fontsize = 8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if savefig:
            fig.savefig(title + '.png')
        
        return fig, ax
    
    def waterfallPlotVertical(pivottable, start = 0, step = 1, num = 2, vert_offset = 0, hor_offset = 0,\
                              xscale = 1, yscale = 1, legendscale = 1, legendunit = '',\
                              xlabel = '', ylabel = '', title = '', savefig = False):
        """
        plot the vertical linecuts of Scan2D.pivottable
        by plotting the horizontal linecuts of Scan2D.pivottable.T
        """
        return ProcessandPlot.waterfallPlotHorizontal(pivottable.T, start, step, num, vert_offset, hor_offset,\
                                                      xscale, yscale, legendscale, legendunit,\
                                                      xlabel, ylabel, title, savefig)