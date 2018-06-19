from os.path import join, basename
import pandas as pd
import matplotlib.pyplot as plt
import os
import nibabel as nb
import re
import nilearn.plotting as plotting
import numpy as np

def get_side(roi_name):
    try:
        if roi_name.endswith('R'):
            return 'Right'
        elif roi_name.endswith('L'):
            return 'Left'
        else:
            return 'Middle'
    except:
        return 'Middle'

def side_remove(roi_name):
    side_removed_roi_name = re.sub(' R| L', '', roi_name)
    return side_removed_roi_name

class jhuLocations:
    def __init__(self):
        self.jhu_label_loc = join(os.environ['FSLDIR'], 
                              'data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz')
        self.jhu_label_img = nb.load(self.jhu_label_loc)
        self.jhu_label_data = self.jhu_label_img.get_data()

        self.jhu_tracts_loc = join(os.environ['FSLDIR'], 'data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz')
        self.jhu_tracts_img = nb.load(self.jhu_tracts_loc)
        self.jhu_tracts_data = self.jhu_tracts_img.get_data()

        self.FA_map_MNI_loc = join(os.environ['FSLDIR'], 'data/standard/FMRIB58_FA_1mm.nii.gz')
        self.FA_mni_img = nb.load(self.FA_map_MNI_loc)
        self.FA_mni_data = self.FA_mni_img.get_data()

class jhuNameLongToShort:
    def __init__(self):
        self.template_shortname_dict = {'Unclassified':'Outside', 'Middle cerebellar peduncle':'MCP', 'Pontine crossing tract (a part of MCP)':'P-MCP', 'Genu of corpus callosum':'G-CC', 'Body of corpus callosum':'B-CC', 'Splenium of corpus callosum':'S-CC', 'Fornix (column and body of fornix)':'FORNIX', 'Corticospinal tract':'CST', 'Medial lemniscus':'ML', 'Inferior cerebellar peduncle':'ICP', 'Superior cerebellar peduncle':'SCP', 'Cerebral peduncle':'CP', 'Anterior limb of internal capsule':'ALIC', 'Posterior limb of internal capsule':'PLIC', 'Retrolenticular part of internal capsule':'RL-ALIC', 'Anterior corona radiata':'ACR', 'Superior corona radiata':'SCR', 'Posterior corona radiata':'PCR', 'Posterior thalamic radiation (include optic radiation)':'PTR', 'Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus)':'SS', 'External capsule':'EC', 'Cingulum (cingulate gyrus)':'CG', 'Cingulum (hippocampus)':'h-CG', 'Fornix (cres) / Stria terminalis (can not be resolved with current resolution)':'FORNIX/STR', 'Superior longitudinal fasciculus':'SLF', 'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule)':'SFOF', 'Uncinate fasciculus':'UF', 'Tapetum':'Tapetum', 'Anterior thalamic radiation':'ATR', 'Corticospinal tract':'CT', 'Forceps major':'F-major', 'Forceps minor':'F-minor', 'Inferior fronto-occipital fasciculus':'IFOF', 'Inferior longitudinal fasciculus':'ILF', 'Superior longitudinal fasciculus (temporal part)':'SLF-t'}

class jhuXmlToDf(jhuLocations):
    def __init__(self):
        jhuLocations.__init__(self)
        self.jhu_label_xml_loc = join(os.environ['FSLDIR'], 
                                      'data/atlases/JHU-labels.xml')
        self.jhu_tract_xml_loc = join(os.environ['FSLDIR'], 
                                      'data/atlases/JHU-tracts.xml')

    def make_jhu_label_df(self):
        with open(self.jhu_label_xml_loc, 'r') as f:
            self.JHU_label_xml = f.read()

        self.jhu_label_num_label_dict = dict(re.findall("index=\"(\d{1,2})\".+>(.+)</label>", 
                                                   self.JHU_label_xml))
        self.jhu_label_df = pd.DataFrame.from_dict(self.jhu_label_num_label_dict,
                                              orient='index', 
                                              columns=['ROI'])
        self.jhu_label_df['Template'] = 'JHU_label'

    def make_jhu_tract_df(self):
        with open(self.jhu_tract_xml_loc, 'r') as f:
            self.JHU_tracts_xml = f.read()

        self.jhu_tract_num_label_dict = dict(re.findall("index=\"(\d{1,2})\".+>(.+)</label>", 
                                                        self.JHU_tracts_xml))
        self.jhu_tract_num_label_zero_match = {}
        self.jhu_tract_num_label_zero_match[0] = 'Outside'
        for key, value in self.jhu_tract_num_label_dict.items():
            self.jhu_tract_num_label_zero_match[int(key)+1] = value
        self.jhu_tract_df = pd.DataFrame.from_dict(self.jhu_tract_num_label_zero_match, 
                                              orient='index', 
                                              columns=['ROI'])  
        self.jhu_tract_df['Template'] = 'JHU_tract'
        
    def concat_jhu_label_and_tract(self):
        self.jhu_df = pd.concat([self.jhu_label_df, 
                                 self.jhu_tract_df])

    def add_side_short_name(self):
        self.jhu_df['side'] = self.jhu_df['ROI'].apply(get_side)
        self.jhu_df['side_removed'] = self.jhu_df['ROI'].apply(side_remove)
        self.jhu_df['short_name'] = self.jhu_df['side_removed'].map(self.template_shortname_dict)

class jhuPlot(jhuXmlToDf, jhuNameLongToShort): 
    def __init__(self, Template, side, short_name):
        jhuXmlToDf.__init__(self)
        jhuXmlToDf.make_jhu_label_df(self)
        jhuXmlToDf.make_jhu_tract_df(self)
        jhuXmlToDf.concat_jhu_label_and_tract(self)
        jhuNameLongToShort.__init__(self)
        jhuXmlToDf.add_side_short_name(self)

        if Template == 'JHU_label':
            template_img = self.jhu_label_img
            template_data = self.jhu_label_data
        else:
            template_img = self.jhu_tracts_img
            template_data = self.jhu_tracts_data

        roi_num = self.jhu_df.groupby(['Template', 'side', 'short_name']).get_group((Template, side, short_name)).index[0]        
        roi_data = np.ma.masked_where(template_data == int(roi_num), template_data).mask.astype(int)
        print(roi_data)
        print(np.sum(roi_data))
        roi_img = nb.Nifti2Image(roi_data, 
                                 affine=template_img.affine)
        long_name = self.jhu_df.groupby(['Template', 'side', 'short_name']).get_group((Template, side, short_name)).ROI.iloc[0]


        plotting.plot_roi(roi_img, 
                          bg_img=self.FA_mni_img, 
                          title=long_name,
                          cmap='autumn')
                          #draw_cross=False)
        plt.show()

def jhu_label_plot(side, short_name):
    jhu = jhuPlot('JHU_label', side, short_name)

def jhu_tract_plot(side, short_name):
    jhu = jhuPlot('JHU_tract', side, short_name)
