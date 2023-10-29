import configparser
import os
class Config():
    def __init__(self) -> None:
        pass
    
    def LoadConf(self, configPath):
        '''读取配置文件'''
        self.conf = configparser.ConfigParser(comment_prefixes='#')
        self.conf.optionxform = lambda option: option
        self.conf.read(configPath) # 文件路径
        self.path = configPath

    def ReadData(self, section, option, fallback = None, type='String'):
        if (type == 'String'):
            data = self.conf.get(section, option, fallback=fallback)
        elif (type == 'Int'):
            data = self.conf.getint(section, option, fallback=fallback)
        elif (type == 'Float'):
            data = self.conf.getfloat(section, option, fallback=fallback)
        elif (type == 'Bool'):   
            data = self.conf.getboolean(section, option, fallback=fallback)
        return data
    
    def WriteData(self, section, option, value):
        if(not self.conf.has_section(section)):
            self.conf.add_section(section)
        self.conf.set(section, option, str(value)) 
        self.AutoSave()
        
    def AutoSave(self):
        f = open(self.path,'w')
        self.conf.write(f)    
        f.close()
        
        
if __name__=='__main__':
    pass