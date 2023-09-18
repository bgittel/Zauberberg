import os
import json
from Zauber_pass5 import df_line
import calendar
import time
current_GMT = time.gmtime()
DIRECTORY_INPUT = '/home/bgittel/Weimar_Prosa'
#DIRECTORY_INPUT = '/home/tneitzke/mona_pipe_b/pipy-public/files_to_process'
DIRECTORY_OUTPUT = '/home/bgittel/zauber_results_pass5_Weimar/'
#DIRECTORY_OUTPUT = '/home/tneitzke/mona_pipe_b/pipy-public/zauber_results_pass/'
all_files = os.listdir(DIRECTORY_OUTPUT)
#print(all_files)
n_fehler = 0

for file in os.listdir(DIRECTORY_INPUT):
     print("Filename:", file)
     file_exists = any(file.startswith(dateiname.split('_result', 1)[0]) and dateiname[-4:] == 'json' for dateiname in all_files)
     print(file, file_exists)
     if file[-3:] == 'txt' and not file_exists:
         try:
            text_file = open(os.path.join(DIRECTORY_INPUT, file))
            text = text_file.read()
            text_file.close()
         except:
            n_fehler = n_fehler + 1
            print("!!! encoding Probleme !!!:" ,file)
            text_file = open(os.path.join(DIRECTORY_INPUT, file), encoding="utf-8", errors='ignore')
            text = text_file.read()
            text_file.close()

         time_stamp = str( calendar.timegm(current_GMT) )
         try:
            df_line_data = df_line(text)
            output_df_line  = df_line_data[0]
            #print(df_line_data[1])
            output_passages = df_line_data[1]
         except:
            print("Fehler beim Prozessieren des Textes")
            n_fehler = n_fehler + 1
            continue
         with open(DIRECTORY_OUTPUT+file[0:-4]+'_result_satistics'+time_stamp+'.json', 'w') as fp1:
             json.dump(output_df_line, fp1)
         with open(DIRECTORY_OUTPUT+file[0:-4]+'_result_passages'+time_stamp+'.json', 'w') as fp2:
             json.dump(output_passages, fp2)
     else:
         print(f"At least one file starting with '{file.split('.', 1)[0]}' exists in the directory. File jumped")
         continue
print("Texte mit Fehler beim prozessieren: ", n_fehler)