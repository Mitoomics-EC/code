import os,sys
import glob
import pandas as pd
from functools import reduce
import numpy as np
import fsd_feature_cal

# super large amount sample prcoess

class FileConcate:
    def __init__(self, sample_path, 
                 target_path = 'fragment_study/mis_10_splitedBy125',
                 specific_path = 'depth_file',
                 file_name = 'depth_stat_LS125_ratio_mis10.csv',
                 tmp_file_name = 'depth_stat_LS125_cunt_mis10.csv',
                 zfile_name = 'depth_stat_LS125_zfsd_mis10.csv'):
        self.sample_path = sample_path
        self.target_path = target_path
        self.specific_path = specific_path
        self.file_name = file_name
        self.tmp_file_name = tmp_file_name
        self.zfile_name = zfile_name
        
    def stat_calculate(self):
        m_list = []
        for s in set(self.sample_name_list):
            col_name_L125 = s + '.mis.10.L125'
            col_name_S125 = s + '.mis.10.S125'

            if col_name_L125 in self.all_data.columns and col_name_S125 in self.all_data.columns:
                try:
                    sub = pd.DataFrame({
                        col_name_L125:
                        self.all_data[col_name_L125] /
                        self.all_data[col_name_S125]
                    })
                    m_list.append(sub)
                except KeyError as e:
                    print(
                        f"KeyError: {e} does not exist in the DataFrame columns"
                    )
            else:
                print(
                    f"Columns {col_name_L125} and/or {col_name_S125} do not exist in the DataFrame"
                )

        if m_list:
            rta = pd.concat(m_list, axis=1)
            rta = rta.round(3)
            rta.to_csv(self.file_save_path, index=False)
            print('results have been saved at ' + self.file_save_path)
            zrta = rta.replace([np.inf, -np.inf], np.nan)
            #zrta = zrta.fillna(method='ffill')
            zrta = zrta.ffill()
            #zrta = zrta.fillna(method='bfill')
            zrta = zrta.bfill()
            zrta = (zrta - zrta.mean()) / zrta.std()
            zrta = zrta.round(3)
            zrta.to_csv(self.zsave_path, index=False)
            print('z-score done,file saved!')

    def file_concate(self):
        specific_file_path = os.path.join(self.sample_path, self.target_path,
                                          self.specific_path, '*.mis.10.*.sam.txt')
        specific_file_list = glob.glob(specific_file_path)
        total_file = len(specific_file_list)
        print(f'{total_file} target file list have been created!')
        print('start concate file...')
        t = 0
        self.sample_name_list = []
        merge_list = []
        merge_dic = {}
        for n in specific_file_list:
            if os.path.getsize(n) > 0:
                sample_name = n.split('/')[-1].split('.')[0]
                col_head = n.split('/')[-1].rsplit('.', 2)[0]
                df = pd.read_csv(n,
                                 sep='\t',
                                 header=None,
                                 index_col=False)
                df1 = pd.DataFrame(df.iloc[16569:, 2]).reset_index()
                df.iloc[0:550, 2] = df.iloc[0:550, 2] + df1[2]
                merge_dic[sample_name] = df.iloc[0:16569, 1:]
                head = ['pos', col_head]
                merge_dic[sample_name].columns = head
                merge_list.append(merge_dic[sample_name])
                t += 1
                self.sample_name_list.append(sample_name)
        self.all_data = reduce(
            lambda left, right: pd.merge(left, right, on='pos'),
            merge_list)
        self.tmp_file_save_path = os.path.join(self.sample_path, self.target_path,
                                          self.tmp_file_name)
        self.all_data.to_csv(self.tmp_file_save_path, index=False)
        if t == total_file:
            print('Concated done!')
            print('start calculate fsd...')
        self.file_save_path = os.path.join(self.sample_path, self.target_path,
                                           self.file_name)
        self.zsave_path = os.path.join(self.sample_path, self.target_path,
                                           self.zfile_name)

        self.stat_calculate()
        

    def runAll(self):
        self.file_concate()
        fsd_feature_get = fsd_feature_cal.ZFSD_Feature_Cal(self.sample_path)
        fsd_feature_get.run()

if __name__ == '__main__':
    sample_path = sys.argv[1]
    concating_file = FileConcate(sample_path)
    concating_file.runAll()
