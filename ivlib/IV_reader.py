"""
作成20240521

Dataについて
Example
IG88--- IG88_C-V Sweep [_ 2 -17_Subsite_0__Subsite_(57) ; 2022-07-06 11_53_44 AM].csv
        ...
     |- IG88_Generic C-f [_ 0 -2_Subsite_0__Subsite_(4) ; 2022-07-06 11_21_58 AM].csv
        ...
     |- IG88_I_V Sweep [_ 1 0_Subsite_0__Subsite_(21) ; 2022-07-06 11_30_17 AM].csv
        ...
今回利用するデータはI_V
I_Vのファイルの中身
ヘッダー : 1行目から248行目まで
コラム名 : 249行目 列名 : V1, I1

"""

import datetime
import os
from pathlib import Path
from pprint import pprint
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import warnings
warnings.simplefilter('ignore')
# warnings.resetwarnings()
# warnings.simplefilter('ignore', FutureWarning)


class IVReader():
    # 電圧スイープの測定データが入っている。(-5V->0->5V, 5V->0->-5)
    
    def __init__(self, file_folder):
        self.file_folder = Path(file_folder)
        
    def read_file(self, info=True, plot=False):
        self.sorted_files_list = get_IV_files_sort(file_path=self.file_folder, 
                                                   search_suff='*_I_V *.csv')
        
        self.all_data_dict, self.select_dict, self.index_dict=read_select(self.sorted_files_list, 
                                                                        sig=1, 
                                                                        info=True, 
                                                                        plot=False)  
    def each_plots(self,all_or_select='select', separater=5):
        if all_or_select == 'select':
            df_3plots(df_lists=self.select_dict["df"],
                      name_lists=self.select_dict["fname"], 
                      comment_lists=None,separator=separater)
            
        else:
            df_3plots(df_lists=all_data_dict["df"],
                      name_lists=all_data_dict["fname"], 
                      comment_lists=all_data_dict["judge"],
                      separator=separater)
            
    def one_fig_plots(self, all_or_select='select', separater=5, save=False):
        if all_or_select == 'select':
            df_multi_3plots(df_lists=self.select_dict["df"], 
                            name_lists=self.select_dict["fname"], 
                            comment_lists=None, sig=1,
                            separator=separater, save=save)
            
        else:
            df_multi_3plots(df_lists=self.all_data_dict["df"], 
                            name_lists=self.all_data_dict["fname"], 
                            comment_lists=self.all_data_dict["judge"], 
                            sig=1, separator=separater, save=save)




# Classを構成する部品

def get_IV_files_sort(file_path, search_suff='*_I_V *.csv'):
    # case: C-f >> search_suff='*C-f *.csv'
    # case: _C_V >> search_suff='*_C_V *.csv'
    
    # example
    # sfile = get_IV_files_sort(file_path=r"schottky\Data\20240404\IGB97", 
    #                         search_suff='*_I_V *.csv')

    
    file_p = Path(file_path) 
    file_p_lists = list(file_p.glob(search_suff)) 
    
    sorted_files_ = _file_num_sort(file_p_lists)
    # sorted_files_ = _file_datetime_sort(file_p_lists)
   
    return sorted_files_
    
def _extract_file_number(filename):
    def extract_subsite_from_filename(filename):
        # Example:
        # filename = 'IGB97_I_V Sweep5V [_ 0 -11_Subsite_0__Subsite_(45) ; 2024-04-04 9_45_46 AM].csv'
        # >>> '_ 0 -11_Subsite_0__Subsite_(45) ; 2024-04-04 9_45_46 AM'
        
        # ファイル名から[]内の文字列を取得する正規表現
        pattern = r"\[(.*?)\]"
        match = re.search(pattern, filename)

        if match:
            # []内の文字列を返す
            return match.group(1)
        else:
            # ファイル名が規則に合致しない場合はNoneを返す
            return None
        
    def get_datetime_from_filename(word):
        # Example
        # word = '_ 0 -11_Subsite_0__Subsite_(45) ; 2024-04-04 9_45_46 AM'
        # >>> datetime.datetime(2024, 4, 4, 9, 45, 46)
        
        if word is None:
            return None
        
        # ファイル名から日時部分を取得する正規表現
        pattern = r"; (.*)"
        match = re.search(pattern, word)

        if match:
            # 日時部分を文字列からdatetime型に変換
            datetime_str = match.group(1)
            datetime_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S %p")
            return datetime_obj
        else:
            # 日時部分がなければNoneを返す
            return None  
      
    extract1 = extract_subsite_from_filename(filename=filename)
    extract2 = get_datetime_from_filename(word=extract1)
    
    return extract1 ,extract2

def extract_xy_num(text):
    """
    text = "_ -1 -14_Subsite_0__Subsite_(15) ; 2024-05-13 11_14_45 AM"
    text ='_ 0 -11_Subsite_0__Subsite_(45) ; 2024-04-04 9_45_46 AM'

    Args:
    text (str): 抽出対象のテキスト

    Returns:
    dict: 抽出したデータ ({'x': -1, 'y': -14}) または None (抽出失敗)

    Examples:
    res = extract_xy('_ 0 -11_Subsite_0__Subsite_(45) ; 2024-04-04 9_45_46 AM')
    print(res['x'],res['y'])
    """
    pattern = r"-?\d+ -?\d+"
    matches = re.findall(pattern, text)[0]
    pattern2 = r"\((.*?)\)"
    matches2 = re.search(pattern2, text)

    if matches:
        num = matches.split(' ')
        xx = int(num[0])
        yy = int(num[1])
        fnum = int(matches2.group(1))
        # print(num)
        # print(xx,yy)
        return {'x':xx, 'y':yy, 'fnum':fnum}
    
    else:
        return {'x':np.nan, 'y':np.nan,'fnum':np.nan}



def _file_datetime_sort(file_list):
    sorted_filenames = sorted(file_list, key=lambda filename: _extract_file_number(str(filename))[1])
    # print(sorted_filenames)
    
    return sorted_filenames


def _file_num_sort(file_list):
    
    def _get_str_num_from_filename(word):
        pattern = r"\((.*?)\)"
        match = re.search(pattern, word)
        # print(match.group(1))
        
        if match:
            num_str =  match.group(1)
            num_int = int(num_str)
            return num_int
        else:
            return None
        
    sorted_filenames = sorted(file_list, key=lambda filename: _get_str_num_from_filename(str(filename)))
    # print(sorted_filenames)
    
    return sorted_filenames


def static_inf(sdata, info=True):
    
    medi_ = np.median(sdata)
    mean_ = np.mean(sdata)
    std_= np.std(sdata)
    cv_ = std_/mean_
    skew_ = sp.stats.skew(sdata)
    kurtosis_ =  sp.stats.kurtosis(sdata, bias=False)
    asymetric_ = medi_/ mean_
    # 変動係数(CV): 標準偏差を平均値で割ったもの。 
    # median =< mean-std とする　(mean-std)/mean = 1 - cv >= median/mean
    # median/mean は変動係数を用いて表すこともできる
   
    info_dict = {'median':medi_, 'mean':mean_, 'std':std_, 
                 'cv':cv_, 'sk':skew_,'kt':kurtosis_,
                 'asymetric':asymetric_}
    
    if info:
        for k, v in info_dict.items():
            print(k, v)
        
    return info_dict

def c_dif(xdata,ydata,flag='g'):
    # return dy/dx, dx, dy
    # data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # c_dif(data,data,'d')
    # array([1., 1., 1., 1., 1., 1., 1., 1.]))

    if flag == 'g': #lenght: len(xdata)
        dy = np.gradient(ydata)      
        dx = np.gradient(xdata)   
    elif flag == 'd' :  # lenght: len(xdata)-1
        dy = np.diff(ydata)      
        dx = np.diff(xdata)   
    
    return   dy/dx, dx, dy


def _divide_df(df_data, target_col='V1', target=5):
    # 検索対象の値のインデックスを取得
    try:
        index = df_data[target_col].index[df_data[target_col] == target].tolist()[0]
    # print(index)
    # 分割後のDataFrameを作成
    except:
        index = len(df_data[target_col]) //2
        
    # df_after = df_data.copy()
    # df_before = df_data.copy() 
    # df_after = df_after.iloc[index+1:]
    # df_before = df_before.iloc[:index+1]
    df_after = df_data.iloc[index+1:]
    df_before = df_data.iloc[:index+1]
    # # 分割結果を出力
    # print("インデックス以降:")
    # print(df_after)
    # print("インデックス以前:")
    # print(df_before)
    return index, df_before, df_after

def df_info(df_data, abs=True, cal_col='I1',info=False):
    
    _, df_f, df_r = _divide_df(df_data,target_col='V1', target=5)
    
    if abs:
        stif = static_inf(np.abs(df_f[cal_col].values),info=info)
        stir = static_inf(np.abs(df_r[cal_col].values),info=info)
    
    else:
        stif = static_inf(df_f[cal_col],info=info)
        stir = static_inf(df_r[cal_col],info=info)
    
    return stif, stir


def read_df(file):
    df_temp = pd.read_csv(file, header=248, usecols=['V1','I1'])
    
    return df_temp   


def read_select(file_lists, sig=-1, info=True, plot=True):
    # Select ルール
    # 測定の行き返りデータのうち片方でも以下の条件に当てはまる場合
    # 強度の絶対値の平均値が
    # 5.0e-9以下---> noise
    # 6e-9以上---> contact
    
    All_df_list = []
    file_name_list = []
    select_ind_list = []
    noise_ind_list = []
    contact_ind_list = []
    
    select_df_list = []
    
    select_filename_list = []
    evaluation_list = []
    
    file_serise_name = Path(file_lists[0]).stem.split(' ')[0]
    
    for i , fn in enumerate(file_lists):
        i2 = Path(fn).stem
        
        # ファイルサイズが0のものが存在するために追加
        if os.path.getsize(fn) == 0:
            print(f"No data: {fn.name}")
            df_temp = pd.DataFrame({'V1':[],'I1':[]})
            
        else:
            df_temp = pd.read_csv(fn, header=248, usecols=['V1','I1'])

        stif,stir= df_info(df_temp, abs=True, cal_col='I1', info=False)
        
        if stif['mean'] < 5.0e-9 or stir['mean'] < 5.0e-9 or stir['mean'] is np.nan:
            # print('noise')
            evaluation_val = 'n'
            noise_ind_list.append(i)
            
        # ファイルサイズが0KBのものはContactに入れる   
        elif stif['mean'] > 6.0e-6 or stir['mean'] > 6.0e-6 or len(df_temp['V1'].values)==0:
            # print('Contact')
            evaluation_val = 'c'
            contact_ind_list.append(i)
            
        else:
            # print('select')
            evaluation_val = 's'
            select_ind_list.append(i)
            select_df_list.append(df_temp)
            select_filename_list.append(i2)
            
        file_name_list.append(i2)
        All_df_list.append(df_temp)
        evaluation_list.append(evaluation_val)
        
        if plot:
            _, df_f, df_r = _divide_df(df_temp)

            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            
            ax1.plot(sig*df_f['V1'],sig*df_f['I1'],
                    label=f"ave:{stif['mean']:.2e}")
            ax1.plot(sig*df_r['V1'],sig*df_r['I1'],
                    label=f"ave:{stir['mean']:.2e}")
            ax2.plot(sig*df_f['V1'],sig*df_f['I1'],
                    label=f"ave:{stif['mean']:.2e}")
            ax2.plot(sig*df_r['V1'],sig*df_r['I1'],
                    label=f"ave:{stir['mean']:.2e}")
            ax2.set_yscale('log')
            ax1.grid()
            ax2.grid()
            ax1.legend()
            ax2.legend()
            fig.suptitle(f'{fn.name}')
            plt.show()
        
    if info:
        print(file_serise_name)
        print(f'All: {len(All_df_list)}')
        print(f'Select: {len(select_ind_list)}')
        print(f'Contact:{len(contact_ind_list)}')
        print(f'Noise:{len(noise_ind_list)}')
    
    index_dict = {'select':select_ind_list,
                  'contact':contact_ind_list,
                  'noise':noise_ind_list}
    
    all_data_dict = {'fname': file_name_list,
                 'judge': evaluation_list,
                 'df':All_df_list}
    
    select_dict = {'fname': select_filename_list,
                  'df' : select_df_list}
    
    return all_data_dict, select_dict, index_dict

def IV_multi_plots(df_lists, name_lists, coment_lists=None, sig=-1, Log=True, title='Plots', separator=5, nrows=None, ncols=3, save=False, plot_fig_step=1):
    """Multiplot
    
    """
    all_data = len(df_lists)
    total = len(df_lists[::plot_fig_step])
    
    if nrows == None:
        nrows = (total + ncols -1)//ncols

    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.8,nrows*3.6), squeeze=False, tight_layout=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*5), squeeze=False, tight_layout=True)
    
    fig.suptitle(title,y=1)

    ax_list=[]    
    for i in range(nrows):
        for j in range(ncols):
            ax_list.append(ax[i,j])
            

    for  idx, (idf, ina) in enumerate(zip(df_lists[::plot_fig_step],name_lists[::plot_fig_step])): 
        _, df_f_, df_r_ =_divide_df(idf, target_col='V1', target=separator)
        
        ax_list[idx].set_title(f'{ina}' ,fontsize=9) 
        ax_list[idx].plot(sig*df_f_['V1'],sig*df_f_['I1'],label=f"data1")
        ax_list[idx].plot(sig*df_r_['V1'],sig*df_r_['I1'],label=f"data2")
        if Log:
            ax_list[idx].set_yscale('log')
                
        ax_list[idx].grid()
        if coment_lists is not None:
            comment = coment_lists[idx]
            ax_list[idx].legend(title=f'{comment}')
        
 
    if len(ax_list) != total:
        for ij in range(len(ax_list)-total):
            newi= ij + total
            ax_list[newi].axis("off")

    plt.tight_layout()
    
    if save:
        filename = re_replace(title)
        plt.savefig(f'{filename}.png')
        
    plt.show() 

def df_multi_2plots(df_lists, name_lists, comment_lists=None, sig=1, title=None, separator=5, nrows=None, ncols=2, save=False, plot_fig_step=1):
    """Multiplot
    Normal , log plot
    
    """
    all_data = len(df_lists)*2
    total = len(df_lists[::plot_fig_step])*2
    
    if nrows == None:
        nrows = (total + ncols -1)//ncols
        
    if title == None:
        title = name_lists[0].split(' ')[0]
        
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.8,nrows*3.6), squeeze=False, tight_layout=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*5), squeeze=False, tight_layout=True)
    
    fig.suptitle(title,y=1)

    ax_list=[]    
    for i in range(nrows):
        for j in range(ncols):
            ax_list.append(ax[i,j])
            
    for  idx, (idf, ina) in enumerate(zip(df_lists[::plot_fig_step],name_lists[::plot_fig_step])): 
        _, df_f_, df_r_ =_divide_df(idf, target_col='V1', target=separator)
        idx2 = idx *2
        
        extract1 ,extract2 = _extract_file_number(ina)
        cord = extract_xy_num(extract1)
        
        ax_list[idx2].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})') 
        ax_list[idx2].plot(sig*df_f_['V1'],sig*df_f_['I1'],label=f"data1")
        ax_list[idx2].plot(sig*df_r_['V1'],sig*df_r_['I1'],label=f"data2")
        ax_list[idx2].grid()
        
        ax_list[idx2+1].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        ax_list[idx2+1].plot(sig*df_f_['V1'],np.abs(sig*df_f_['I1'].values),label=f"data1")
        ax_list[idx2+1].plot(sig*df_r_['V1'],np.abs(sig*df_r_['I1'].values),label=f"data2")
        ax_list[idx2+1].set_yscale('log')
        ax_list[idx2+1].grid()        

        if coment_lists is not None:
            comment = coment_lists[idx]
            ax_list[idx2].legend(title=f'{comment}')
            ax_list[idx2+1].legend(title=f'{comment}')
 
    if len(ax_list) != total:
        for ij in range(len(ax_list)-total):
            newi= ij + total
            ax_list[newi].axis("off")

    plt.tight_layout()
    
    if save:
        filename = re_replace(title)
        plt.savefig(f'{filename}.png')
        
    plt.show() 

def df_multi_3plots(df_lists, name_lists, comment_lists=None, 
                    sig=1, title=None, separator=5, nrows=None, ncols=3, 
                    save=False, plot_fig_step=1):
    """Multiplot
    Normal , log plot, log vs sqrt V
    
    """
    all_data = len(df_lists)*3
    total = len(df_lists[::plot_fig_step])*3
    
    if nrows == None:
        nrows = (total + ncols -1)//ncols

    if title == None:
        title = name_lists[0].split(' ')[0]
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.8,nrows*3.6), squeeze=False, tight_layout=True)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6,nrows*5), squeeze=False, tight_layout=True)
    
    fig.suptitle(title,y=1)

    ax_list=[]    
    for i in range(nrows):
        for j in range(ncols):
            ax_list.append(ax[i,j])
            

    for  idx, (idf, ina) in enumerate(zip(df_lists[::plot_fig_step],name_lists[::plot_fig_step])): 
        _, df_f_, df_r_ =_divide_df(idf, target_col='V1', target=separator)
        idx2 = idx *3
        
        extract1 ,extract2 = _extract_file_number(ina)
        cord = extract_xy_num(extract1)
        
        ax_list[idx2].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        ax_list[idx2].plot(sig*df_f_['V1'],sig*df_f_['I1'],label=f"data1")
        ax_list[idx2].plot(sig*df_r_['V1'],sig*df_r_['I1'],label=f"data2")
        ax_list[idx2].set_xlabel('Voltage')
        ax_list[idx2].set_ylabel('Current')
        ax_list[idx2].grid()
        
    
        ax_list[idx2+1].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        ax_list[idx2+1].plot(sig*df_f_['V1'],np.abs(sig*df_f_['I1'].values),label=f"data1")
        ax_list[idx2+1].plot(sig*df_r_['V1'],np.abs(sig*df_r_['I1'].values),label=f"data2")
        ax_list[idx2+1].set_yscale('log')
        ax_list[idx2+1].set_xlabel('Voltage')
        ax_list[idx2+1].set_ylabel('ln(Current)')
        ax_list[idx2+1].grid()   
        
        ax_list[idx2+2].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        ax_list[idx2+2].plot(np.sqrt(np.abs(sig*df_f_['V1'].values)),np.abs(sig*df_f_['I1'].values),label=f"data1")
        ax_list[idx2+2].plot(np.sqrt(np.abs(sig*df_r_['V1'].values)),np.abs(sig*df_r_['I1'].values),label=f"data2")
        ax_list[idx2+2].set_yscale('log')
        ax_list[idx2+2].set_xlabel('Voltage$^{1/2}$')
        ax_list[idx2+2].set_ylabel('ln(Current)')
        ax_list[idx2+2].grid()        

        if comment_lists is not None:
            comment = comment_lists[idx]
            ax_list[idx2].legend(title=f'{comment}')
            ax_list[idx2+1].legend(title=f'{comment}')
            ax_list[idx2+2].legend(title=f'{comment}')
 
    if len(ax_list) != total:
        for ij in range(len(ax_list)-total):
            newi= ij + total
            ax_list[newi].axis("off")

    plt.tight_layout()
    
    if save:
        filename = re_replace(title)
        plt.savefig(f'{filename}.png')
        
    plt.show() 


def df_3plots(df_lists, name_lists, 
              comment_lists=None, sig=1, separator=5):
    """
    Normal, log plot, log vs sqrt V
    
    """
    print(name_lists[0].split(' ')[0])
    
    for  idx, (idf, ina) in enumerate(zip(df_lists,name_lists)): 
        _, df_f_, df_r_ = _divide_df(idf, target_col='V1', target=separator)

        extract1, extract2 = _extract_file_number(ina)
        cord = extract_xy_num(extract1)
        
        fig, axs = plt.subplots(1,3,figsize=(18,5), squeeze=False, tight_layout=True)

        axs[0,0].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        axs[0,0].plot(sig*df_f_['V1'],sig*df_f_['I1'],label=f"data1")
        axs[0,0].plot(sig*df_r_['V1'],sig*df_r_['I1'],label=f"data2")
        axs[0,0].set_xlabel('Voltage')
        axs[0,0].set_ylabel('Current')
        axs[0,0].grid()
        
        axs[0,1].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        axs[0,1].plot(sig*df_f_['V1'],np.abs(sig*df_f_['I1'].values),label=f"data1")
        axs[0,1].plot(sig*df_r_['V1'],np.abs(sig*df_r_['I1'].values),label=f"data2")
        axs[0,1].set_yscale('log')
        axs[0,1].set_xlabel('Voltage')
        axs[0,1].set_ylabel('ln(Current)')
        axs[0,1].grid()   
        
        axs[0,2].set_title(f'Index:{idx}, No:{cord["fnum"]}, (x,y)=({cord["x"]},{cord["y"]})')
        axs[0,2].plot(np.sqrt(np.abs(sig*df_f_['V1'].values)),np.abs(sig*df_f_['I1'].values),label=f"data1")
        axs[0,2].plot(np.sqrt(np.abs(sig*df_r_['V1'].values)),np.abs(sig*df_r_['I1'].values),label=f"data2")
        axs[0,2].set_yscale('log')
        axs[0,2].set_xlabel('Voltage$^{1/2}$')
        axs[0,2].set_ylabel('ln(Current)')
        axs[0,2].grid()        

        if comment_lists is not None:
            comment = comment_lists[idx]
            axs[0,0].legend(title=f'{comment}')
            axs[0,1].legend(title=f'{comment}')
            axs[0,2].legend(title=f'{comment}')
            
        plt.tight_layout()
        plt.show() 
     
    
def re_replace(text):
    """Remove special symbols with regular expressions 

    Args:
        text (str): text
    Returns:
        str: Text with special symbols removed
    Examples:
        text = '4 inch $\phi$=0'
        re_replace(text)
        >>> '4_inch_phi0
    Ref:
        https://qiita.com/ganyariya/items/42fc0ed3dcebecb6b117 
    """
    # code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')

    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~]')
    cleaned_text = code_regex.sub('', text).replace(' ', '_')
    # print(cleaned_text)

    return cleaned_text
   
def data_save_name(file_folder):
    file_p = Path(file_folder)
    s_name = f'{file_p.parents[0].stem}_{file_p.stem}'
    return s_name

def fig3plot(xdata,ydata,x2data,y2data,comment=None):
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # ax_list[idx2].set_title(f'{ina}' ,fontsize=9) 
    # axs[0].set_title(f'{cord}' ,fontsize=9) 
    axs[0].plot(xdata,ydata,label=f"data1")
    axs[0].plot(x2data,y2data,label=f"data2")
    axs[0].set_xlabel('Voltage')
    axs[0].set_ylabel('Current')
    axs[0].grid()
    
    # ax_list[idx2+1].set_title(f'{ina}' ,fontsize=9) 
    # axs[1].set_title(f'{cord}' ,fontsize=9) 
    axs[1].plot(xdata,np.abs(ydata),label=f"data1")
    axs[1].plot(x2data,np.abs(y2data),label=f"data2")
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Voltage')
    axs[1].set_ylabel('ln(Current)')
    axs[1].grid()   
    
    # axs[2].set_title(f'{cord}' ,fontsize=9) 
    axs[2].plot(np.sqrt(np.abs(xdata)),np.abs(ydata),label=f"data1")
    axs[2].plot(np.sqrt(np.abs(x2data)),np.abs(y2data),label=f"data2")
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Voltage$^{1/2}$')
    axs[2].set_ylabel('ln(Current)')
    axs[2].grid()        

    if comment is None:
        axs[0].legend(title=f'{comment}')
        axs[1].legend(title=f'{comment}')
        axs[2].legend(title=f'{comment}')

    # else:
    #     comment='data'
    plt.tight_layout()
        
    # if save:
    #     filename = re_replace(comment)
    #     plt.savefig(f'{comment}.png', dpi=500)
            
    plt.show() 
   
if __name__ == "__main__":
    pass