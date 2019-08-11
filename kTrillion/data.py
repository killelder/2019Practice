import pandas as pd
import os
TF_TICK = 0
TF_SECOND = 1
TF_MINUTE = 2
TF_HOUR = 3
TF_DAY = 4
TF_WEEK = 5
TF_MON = 6
TF_YEAR = 7

class data():
    def ___init__(self):
        
        self.o = None #Open 開
        self.h = None #High 高
        self.l = None #Low  低
        self.c = None #Close 收
        self.v = None #Volume 成交量
        self.t = None #TimeSeries 時間
        self.tflag = 0 #TimeFlag 由小往大可以延伸, 但不能向下延伸
    
    def load_data(self, path):
        """
            csv, numpy, pandas
            要把data換成西元
            把data不要有多餘的空白, 至少time那邊不要有
        """
        def dateparse(d):
            #dt = d + " " + t
            print("?????", d)
            return pd.datetime.strptime(d, '%Y/%m/%d')
            
        #dateparse = lambda dates: pd.datetime.strptime(dates,'%y/%m/%d')
        df = pd.read_csv(path)
        #print(df['time'])
        #df['time'] = df['time'].str.replace(" ", "")
        df['time'] = pd.to_datetime(df["time"], format="%Y/%m/%d", errors='coerce')
        #df = pd.read_csv(path, parse_dates=['time'], index_col='time', date_parser=dateparse)
        print(df["time"])
        pass
        
    def to_time_flag(self):
        """
            根據timeflag延伸
        """
        pass
        
    def Oper(self, func):
        """
            Operation on data
        """
        pass
        
def sort_data():
    """
        用來轉換那些有問題的data
    """
    for root, dirs, files in os.walk("./data"):
        for f in files:
            if ".csv" in f:
                fp = open(os.path.join(root,f), "r")
                buf = fp.readline()
                newfp = open("./pdata/" + f, "w")
                newfp.write(buf)
                for buf in fp:
                    
                    buf2 = buf.split(",")
                    newfp.write(str(int(buf2[0].split("/")[0])+1911)+"/"+buf2[0].split("/")[1]+"/"+buf2[0].split("/")[2]+",")
                    for j in range(1, len(buf2)):
                        if j == len(buf2)-1:
                            newfp.write(buf2[j])
                        else:
                            newfp.write(buf2[j]+",")
if __name__ == "__main__":
    #sort_data()
    d_ = data()
    d_.load_data("./data/1101.csv")