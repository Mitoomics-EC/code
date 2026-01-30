import os, sys
import glob
import pandas as pd
from median_cal import median_isnertsize
from functools import reduce


class FileConcate:
    def __init__(self,
                 sample_path,
                 file_name,
                 target_path=None,
                 specific_path=None):
        self.sample_path = sample_path
        self.target_path = target_path
        self.specific_path = specific_path
        self.file_name = file_name
        #self.result_path = result_path

    def file_concate(self):
        #self.create_insertsize_mtr()
        sam_path = os.path.join(self.sample_path,
                                '*/*.mis.10.insertsize.txt')
        specific_file_list = glob.glob(sam_path)

        total_file = len(specific_file_list)
        print(f'{total_file} target file list have been created!')
        print('start concate file...')
        t = 0
        mlist = [pd.DataFrame({'insert_size': range(2001)})]
        for n in specific_file_list:
            if os.path.getsize(n)>0:
                sample_name = n.split('/')[-1].split('.')[0]
                dfn = pd.read_csv(
                    n, sep='\t', skiprows=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                    ]).rename(columns={'All_Reads.fr_count': sample_name})
                mlist.append(dfn)
                t += 1
        if t == total_file:
            print('merge list have been created!')
            print("starting concat all files")
        #merge_list.append(df)
        #dfm = reduce(lambda x,y:pd.merge(x, y, on='insert_size', how='outer'),merge_list).fillna(0)
        dfm = reduce(lambda x,y:pd.merge(x, y, on='insert_size', how='outer'),mlist).fillna(0)
        dfm = dfm.sort_values(by="insert_size").reset_index()
        dfm = dfm.drop(columns=["index"])
        outpath = os.path.join(
            self.sample_path,
            "fragment_study",self.specific_path)
        os.makedirs(outpath, exist_ok=True)
        self.result_file = os.path.join(outpath,
                                        self.file_name)
        dfm.to_csv(self.result_file, index=False)
        median_isnertsize(
            inpath=outpath,
            infile=self.file_name,
            outfile1="mis10_insertsize_cufrequency.csv",
            outfile2="mis10_insertsize_cumfrequency50%_insertsize.csv")

    def runAll(self):
        #self.create_insertsize_mtr()
        self.file_concate()


if __name__ == '__main__':
    sample_path = sys.argv[1]
    target_path = 'fragment_study'
    #specific_path = 'mis_10_insertsize1'
    #result_path = '/mnt/hdd/usr/jiaohuanmin/home/jiaohm/data/liuy/fragment_study/LLI/insertsize'
    #file_name = 'mis_10_tRNA_insertsize_mtr_231215.csv'
    file_concate = FileConcate(
        sample_path,
        file_name='mis_10_insertsize_mtr.csv',
        target_path='fragment_study',
        specific_path='mis_10_insertsize')
    file_concate.runAll()