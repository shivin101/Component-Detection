
import configparser

# CREATE OBJECT
config_file = configparser.ConfigParser()

# ADD FileConfigs SECTION
config_file.add_section("FileConfig")

# ADD SETTINGS TO FileConfigs SECTION
config_file.set("FileConfig", "root", '/home/kohyoung/edge_detection/Data/Mobile - Black board - Shield - Narrow gaps 5')
config_file.set("FileConfig", "model_root", '/home/kohyoung/edge_detection/bcdn/final-model/bdcn_pretrained_on_bsds500.pth')
config_file.set("FileConfig", "bcdn_path", 'home/kohyoung/edge_detection/bcdn/BDCN')
config_file.set("FileConfig", "img_dir", '/bmps/')
config_file.set("FileConfig", "height_dir", '/height maps/RAW/')
config_file.set("FileConfig", "save_dir", '../Processed_data/')


config_file["PostProcess"]={"root" : '../PcbData/',
"data_id" : 'Mobile - Black board - Shield - Narrow gaps 6',
"session_id" : 7,
"saved_root" :'../Processed_data/'}

# ADD NEW SECTION AND SETTINGS
config_file["Logger"]={
        "LogFilePath":"<Path to log file>",
        "LogFileName" : "<Name of log file>",
        "LogLevel" : "Info"
        }

# SAVE CONFIG FILE
with open(r"configurations.ini", 'w') as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'configurations.ini' created")

# PRINT FILE CONTENT
read_file = open("configurations.ini", "r")
content = read_file.read()
print("Content of the config file are:\n")
print(content)
read_file.flush()
read_file.close()